"""Local-only FastAPI scoring surface. See deployment security limitations."""
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import redis
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from fraud_detection.simulation import monitoring as metrics
from fraud_detection.simulation.decision import BenchmarkModel, DecisionEngine
from fraud_detection.simulation.explanations import explain
from fraud_detection.simulation.review import ReviewFeedback, add_feedback, create_case, get_case, list_cases, save_case
from fraud_detection.simulation.schemas import Authorization, Decision, FraudLabel, Thresholds, identifier, utcnow
from fraud_detection.simulation.state import Conflict, RedisState
from fraud_detection.simulation.transport import KafkaPublisher


def create_app(state=None, engine=None, publisher=None):
    @asynccontextmanager
    async def lifespan(app):
        if state is None:
            client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True,
                                          socket_connect_timeout=2, socket_timeout=2)
            app.state.store = RedisState(client, os.getenv("STATE_NAMESPACE", "bank-demo"))
        else:
            app.state.store = state
        app.state.engine = engine or DecisionEngine(
            BenchmarkModel(os.getenv("MODEL_PATH", "artifacts/model-validation.joblib"),
                           os.getenv("FEATURE_DB", "artifacts-replay/features.sqlite")),
            Thresholds(review=float(os.getenv("REVIEW_THRESHOLD", ".55")), block=float(os.getenv("BLOCK_THRESHOLD", ".98"))),
            blocked_cards=filter(None, os.getenv("BLOCKED_CARD_TOKENS", "").split(",")),
        )
        app.state.publisher = publisher
        bootstrap = os.getenv("KAFKA_BOOTSTRAP")
        if app.state.publisher is None and bootstrap:
            app.state.publisher = KafkaPublisher(bootstrap)
        yield
        producer = getattr(app.state.publisher, "producer", None)
        if producer is not None:
            producer.flush(5)

    app = FastAPI(title="Banking flow SIMULATION — not production", lifespan=lifespan)

    @app.middleware("http")
    async def telemetry(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        if request.url.path == "/score":
            metrics.REQUESTS.labels(str(response.status_code)).inc()
            metrics.LATENCY.observe(time.perf_counter() - start)
        return response

    @app.get("/health/live")
    def live():
        return {"status": "alive", "simulation": True}

    @app.get("/health/ready")
    def ready():
        try:
            app.state.store.ping()
        except redis.RedisError:
            raise HTTPException(503, "State store unavailable")
        return {"status": "ready", "model_version": app.state.engine.model.version,
                "simulation": True, "rules_fallback": app.state.engine.model.version == "unavailable"}

    @app.post("/score", response_model=Decision)
    def score(event: Authorization, x_api_key: str | None = Header(default=None)):
        require_api_key(x_api_key)
        try:
            decision, duplicate = app.state.store.process(event, lambda context: app.state.engine.decide(event, context))
        except Conflict as error:
            raise HTTPException(409, str(error))
        except (redis.RedisError, RuntimeError):
            # No unpersisted fallback decision: caller must retry, preserving idempotence.
            raise HTTPException(503, "State unavailable or contended; retry this same event_id")
        if duplicate:
            metrics.DUPLICATES.inc()
        else:
            metrics.DECISIONS.labels(decision.decision).inc()
        create_case(app.state.store, decision)
        return decision

    def require_api_key(api_key):
        configured = os.getenv("FRAUD_API_KEY")
        if configured and api_key != configured:
            raise HTTPException(401, "Invalid API key")

    @app.get("/decisions/{event_id}/explanation")
    def explanation(event_id: str, x_api_key: str | None = Header(default=None)):
        require_api_key(x_api_key)
        if not event_id.startswith("evt_") or len(event_id) > 80:
            raise HTTPException(404, "Unknown decision")
        store = app.state.store
        try:
            raw = store.client.get(store.prefix + "event:" + event_id) if isinstance(store, RedisState) else store.values.get("event:" + event_id)
        except redis.RedisError:
            raise HTTPException(503, "State store unavailable")
        if raw is None:
            raise HTTPException(404, "Unknown decision")
        cached = json.loads(raw)
        decision = Decision.model_validate(cached["decision"])
        model_effects = None
        event_data = cached.get("event")
        explain_features = getattr(app.state.engine.model, "explain_features", None)
        if event_data and explain_features:
            try:
                model_effects = explain_features(Authorization.model_validate(event_data))
            except (ValueError, OSError):
                model_effects = {"method": "unavailable", "effects": []}
        return explain(decision, model_effects=model_effects,
                       llm_url=os.getenv("LOCAL_LLM_URL") or None,
                       llm_model=os.getenv("LOCAL_LLM_MODEL", "qwen2.5:0.5b"),
                       remote_url=os.getenv("REMOTE_LLM_URL") or None,
                       remote_model=os.getenv("REMOTE_LLM_MODEL") or None,
                       remote_api_key=os.getenv("REMOTE_LLM_API_KEY") or None)

    @app.get("/investigations")
    def investigations(decision: str | None = None, min_score: float | None = None,
                       max_score: float | None = None, status: str | None = None,
                       search: str | None = None, from_time: str | None = None,
                       to_time: str | None = None, offset: int = 0, limit: int = 100,
                       x_api_key: str | None = Header(default=None)):
        require_api_key(x_api_key)
        if decision not in {None, "approve", "manual_review", "decline"}:
            raise HTTPException(422, "Invalid decision filter")
        if status not in {None, "open", "confirmed"}:
            raise HTTPException(422, "Invalid status filter")
        if min_score is not None and not 0 <= min_score <= 1 or max_score is not None and not 0 <= max_score <= 1:
            raise HTTPException(422, "Score filters must be between 0 and 1")
        if min_score is not None and max_score is not None and min_score > max_score:
            raise HTTPException(422, "min_score must not exceed max_score")
        if offset < 0 or not 1 <= limit <= 500:
            raise HTTPException(422, "offset must be non-negative and limit must be between 1 and 500")

        def parse_time(value):
            if not value:
                return None
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                raise HTTPException(422, "Dates must be ISO-8601")
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()

        start_time, end_time = parse_time(from_time), parse_time(to_time)
        if start_time and end_time and start_time > end_time:
            raise HTTPException(422, "from_time must not exceed to_time")
        return list_cases(app.state.store, decision=decision, min_score=min_score,
                          max_score=max_score, status=status, search=search,
                          from_time=start_time, to_time=end_time,
                          offset=offset, limit=limit)

    @app.get("/investigations/{case_id}")
    def investigation(case_id: str, x_api_key: str | None = Header(default=None)):
        require_api_key(x_api_key)
        case = get_case(app.state.store, case_id)
        if case is None:
            raise HTTPException(404, "Unknown investigation")
        return case

    @app.post("/investigations/{case_id}/feedback")
    def feedback(case_id: str, payload: ReviewFeedback, x_api_key: str | None = Header(default=None)):
        require_api_key(x_api_key)
        case = add_feedback(app.state.store, case_id, payload)
        if case is None:
            raise HTTPException(404, "Unknown investigation")
        if payload.verdict in {"confirmed_fraud", "legitimate"} and app.state.publisher is not None:
            outbox = case.setdefault("label_outbox", {})
            event_payload = outbox.get(payload.verdict)
            if event_payload is None:
                decision = Decision.model_validate(case["decision"])
                label_time = max(utcnow(), decision.event_time + timedelta(milliseconds=1))
                label = FraudLabel(
                    event_id=identifier("evt", f"analyst-label:{case_id}:{payload.verdict}"),
                    transaction_id=decision.transaction_id, trace_id=decision.trace_id,
                    event_time=label_time, label_time=label_time,
                    authorization_time=decision.event_time,
                    label=1 if payload.verdict == "confirmed_fraud" else 0,
                    label_type="chargeback" if payload.verdict == "confirmed_fraud" else "historical_confirmation",
                )
                event_payload = label.model_dump(mode="json")
                outbox[payload.verdict] = event_payload
                save_case(app.state.store, case)
            try:
                app.state.publisher.send("fraud-labels", event_payload)
            except Exception:
                raise HTTPException(503, "Feedback saved but label event was not acknowledged; retry the same verdict")
            return {**case, "label_event_id": event_payload["event_id"], "label_event_published": True}
        return case

    @app.get("/analyst", response_class=HTMLResponse)
    def analyst_console():
        return """<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Revue fraude</title>
        <style>body{font:15px system-ui;max-width:1440px;margin:1.5rem auto;padding:0 1rem;background:#f4f1ea;color:#17221b}h1{font:600 28px Georgia,serif}form{display:flex;flex-wrap:wrap;gap:.65rem;align-items:end;padding:1rem 0;border-block:1px solid #b9c3bb}label{display:grid;gap:.25rem;font-size:12px}input,select,button{font:inherit;padding:.45rem .55rem;border:1px solid #829087;border-radius:3px;background:white;color:inherit}button{cursor:pointer;background:#d8e8dc}table{width:100%;border-collapse:collapse;background:white;margin-top:1rem}td,th{padding:.55rem .65rem;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}th{background:#e6ebe6;position:sticky;top:0}td small{display:block;color:#52615a}#status{min-height:1.3em}details{min-width:220px}pre{white-space:pre-wrap;max-width:360px;font:12px ui-monospace,monospace}</style>
        <h1>Dossiers de revue</h1><label>Cle API <input id="key" type="password" placeholder="laisser vide si desactivee"></label>
        <form id="filters"><label>Decision<select name="decision"><option value="">Toutes</option><option>manual_review</option><option>decline</option></select></label>
        <label>Statut<select name="status"><option value="">Tous</option><option>open</option><option>confirmed</option></select></label>
        <label>Score minimum<input name="min_score" type="number" min="0" max="1" step="0.01"></label><label>Score maximum<input name="max_score" type="number" min="0" max="1" step="0.01"></label>
        <label>Depuis<input name="from_time" type="datetime-local"></label><label>Jusqu'a<input name="to_time" type="datetime-local"></label>
        <label>Recherche<input name="search" type="search" placeholder="dossier, motif, modele"></label><button type="submit">Filtrer</button></form>
        <p id="status">Chargement des dossiers…</p><table><thead><tr><th>Dossier / date</th><th>Decision / score</th><th>Modele</th><th>Motifs et explication</th><th>Action analyste</th></tr></thead><tbody id="cases"></tbody></table>
        <script>
        const headers=json=>({'X-API-Key':document.querySelector('#key').value,...(json?{'content-type':'application/json'}:{})});
        function cell(row,value){const td=document.createElement('td');td.textContent=value??'-';row.append(td);return td}
        async function load(){const form=new FormData(document.querySelector('#filters'));const params=new URLSearchParams();for(const [k,v] of form)if(v)params.set(k,v);const response=await fetch('/investigations?'+params,{headers:headers(false)});const status=document.querySelector('#status');const body=document.querySelector('#cases');body.replaceChildren();if(!response.ok){status.textContent='Acces refuse ou filtres invalides';return}const cases=await response.json();status.textContent=cases.length+' dossier(s) affiches';for(const item of cases){const decision=item.decision;const row=document.createElement('tr');cell(row,item.case_id+'\n'+item.created_at);cell(row,decision.decision+'\n'+(decision.risk_score??'-'));cell(row,decision.model_version);const explain=cell(row,decision.reasons.join(', '));const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='Explication modele';details.append(summary);const output=document.createElement('pre');details.append(output);summary.addEventListener('click',async()=>{if(output.textContent)return;const result=await fetch('/decisions/'+encodeURIComponent(decision.request_event_id)+'/explanation',{headers:headers(false)});output.textContent=result.ok?JSON.stringify(await result.json(),null,2):'Explication indisponible'});explain.append(details);const actions=cell(row,'');for(const [verdict,label] of [['confirmed_fraud','Fraude'],['legitimate','Legitime']]){const button=document.createElement('button');button.type='button';button.textContent=label;button.addEventListener('click',()=>send(item.case_id,verdict));actions.append(button)}body.append(row)}}
        async function send(id,verdict){const response=await fetch('/investigations/'+encodeURIComponent(id)+'/feedback',{method:'POST',headers:headers(true),body:JSON.stringify({verdict,analyst:'console',note:''})});document.querySelector('#status').textContent=response.ok?'Verdict enregistre'+(response.headers.get('content-type')?.includes('json')?'':''):'Echec de publication du feedback';load()}
        document.querySelector('#filters').addEventListener('submit',event=>{event.preventDefault();load()});load();</script></html>"""

    @app.get("/metrics")
    def monitoring():
        return Response(generate_latest(metrics.REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()

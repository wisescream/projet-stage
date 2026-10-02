"""Analyst-only explanations; never fed back into the decision engine."""
import json
from urllib.parse import urlparse

import httpx

from fraud_detection.simulation.knowledge import retrieve_context

TEXT = {
    "blocked_card": "Carte présente dans la liste de blocage de simulation.",
    "unusual_amount": "Montant supérieur au seuil de sécurité du prototype.",
    "high_velocity": "Répétition de transactions ou refus récents.",
    "new_device": "Appareil différent du précédent pour ce client.",
    "country_change": "Changement de pays dans une courte fenêtre simulée.",
    "multiple_cards_on_device": "Plusieurs cartes observées sur cet appareil.",
    "three_ds_missing": "Authentification 3-D Secure absente pour un montant élevé.",
    "identity_data_missing": "Données identity IEEE absentes ; signal informatif.",
    "model_unavailable": "Modèle ou features indisponibles : aucune autorisation automatique.",
    "model_block_threshold": "Score au-dessus du seuil de refus.",
    "model_review_threshold": "Score au-dessus du seuil de revue.",
    "anomaly_review_threshold": "Le comportement est atypique par rapport aux références historiques.",
    "anomaly_unavailable": "Le score de détection d'anomalie est indisponible.",
    "below_risk_thresholds": "Aucun seuil ni règle de revue/refus déclenché.",
}


def explain(decision, *, llm_url=None, llm_model="qwen2.5:0.5b",
            remote_url=None, remote_model=None, remote_api_key=None, model_effects=None):
    facts = {"decision": decision.decision, "risk_score": decision.risk_score,
             "anomaly_score": decision.anomaly_score,
             "reasons": [TEXT[r] for r in decision.reasons], "model_version": decision.model_version}
    query = " ".join(facts["reasons"])
    result = {"event_id": decision.request_event_id, "trace_id": decision.trace_id,
              "facts": facts, "explanation": " ".join(facts["reasons"]), "generated_by": "template",
              "analyst_only": True, "requires_human_validation": False,
              "retrieved_context": retrieve_context(query)}
    if model_effects is not None:
        result["model_feature_effects"] = model_effects
    if llm_url:
        parsed = urlparse(llm_url)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "ollama", "host.docker.internal"} or parsed.username:
            raise ValueError("Only an explicitly enabled local LLM endpoint is supported")
        try:
            response = httpx.post(llm_url.rstrip("/") + "/api/generate", timeout=15, trust_env=False, json={
                "model": llm_model, "stream": False, "options": {"temperature": 0, "num_predict": 160},
                "prompt": "Explique en français ces faits uniquement. Ne prends aucune décision, ne déduis aucune culpabilité et n'ajoute aucun fait. " + json.dumps(facts),
            })
            response.raise_for_status()
            text = response.json()["response"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty local model response")
            result.update(explanation=text[:2000], generated_by="local_llm", requires_human_validation=True)
        except (httpx.HTTPError, ValueError, KeyError):
            result["llm_status"] = "unavailable_template_fallback"
    if remote_url:
        parsed = urlparse(remote_url)
        if parsed.scheme != "https" or parsed.username or not remote_api_key:
            raise ValueError("Remote LLM requires an HTTPS URL and an injected API key")
        try:
            response = httpx.post(remote_url.rstrip("/") + "/chat/completions", timeout=20,
                                  trust_env=False, headers={"Authorization": "Bearer " + remote_api_key}, json={
                                      "model": remote_model or "meta-llama/Llama-3.2-3B-Instruct",
                                      "temperature": 0,
                                      "max_tokens": 160,
                                      "messages": [{"role": "system", "content": "Explique uniquement les faits fournis en francais. Ne prends aucune decision et n'ajoute aucun fait."},
                                                   {"role": "user", "content": json.dumps(facts)}],
                                  })
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty remote model response")
            result.update(explanation=text[:2000], generated_by="remote_llm", requires_human_validation=True)
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
            result["llm_status"] = "unavailable_template_fallback"
    return result

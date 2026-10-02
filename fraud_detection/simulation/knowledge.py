"""Small controlled knowledge base for analyst explanations.

This is retrieval-augmented context, not an autonomous decision source. Documents
are versioned in code so an analyst can audit exactly what was retrieved.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str
    title: str
    text: str
    version: str = "2026-09-18"


DOCUMENTS = (
    KnowledgeDocument("policy-amount", "Montant inhabituel", "Un montant eleve doit declencher une verification supplementaire. Il ne suffit pas seul a conclure a une fraude."),
    KnowledgeDocument("policy-velocity", "Velocite", "Une repetition rapide de transactions ou d echecs doit etre revue avec le contexte client et appareil."),
    KnowledgeDocument("policy-3ds", "Authentification 3-D Secure", "Une authentification 3-D Secure absente pour un montant eleve justifie une revue ou une authentification renforcee."),
    KnowledgeDocument("policy-model", "Score du modele", "Le score du modele sert au triage. Une decision de refus automatique doit rester reservee aux seuils et regles fortes."),
)


def retrieve_context(query, limit=3):
    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    ranked = []
    for document in DOCUMENTS:
        tokens = set(re.findall(r"[a-z0-9]+", (document.title + " " + document.text).lower()))
        overlap = len(words & tokens)
        if overlap:
            ranked.append((overlap, document))
    ranked.sort(key=lambda item: (-item[0], item[1].document_id))
    return [{"document_id": document.document_id, "title": document.title,
             "version": document.version, "text": document.text, "match_count": overlap}
            for overlap, document in ranked[:limit]]
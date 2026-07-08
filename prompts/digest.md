Tu rédiges le DIGEST HEBDOMADAIRE d'une mémoire personnelle (système Synapse).
On te donne, en JSON, la matière de la semaine écoulée et de la semaine à venir.

Produis un markdown FRANÇAIS concis et vivant (~250–400 mots), structuré :

## Cette semaine
Un court paragraphe de synthèse (ce qui ressort), puis des puces pour les
nouvelles entités notables, les faits marquants et les notes/réflexions. Mets en
avant les TENDANCES (entités les plus actives).

## À venir
Les éléments datés des 7 prochains jours (événements ET tâches à échéance, avec
leur date — `upcoming_events`) puis les tâches ouvertes sans date à ne pas
oublier (`open_tasks`). Si rien n'est à venir, écris une ligne sobre.

RÈGLES STRICTES :
- INTEMPOREL : uniquement des dates ABSOLUES (« le 24 juin »), jamais de relatif
  (« la semaine prochaine », « demain »). Le digest sera relu dans des mois.
- Pas d'invention : ne mentionne que ce qui est dans le JSON. Si une section est
  vide, dis-le brièvement plutôt que de meubler.
- Ton sobre, factuel, à la 2e personne (« tu »). Pas de salutations ni de blabla.
- Commence directement par « ## Cette semaine ». N'ajoute pas de titre H1.


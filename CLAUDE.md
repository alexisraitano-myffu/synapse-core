# synapse-core

Le cerveau compilé unique de Synapse — **complet depuis T5 (SYN-114)** : embeddings, stockage (schéma SQLite + sqlite-vec), classification + routing, decay, resummary/synthèse projet, digest hebdo, ressources web, sync P2P — écrit une fois en Rust et consommé partout : backend FastAPI Mac via PyO3, apps mobiles via UniFFI. Une seule implémentation, zéro divergence de logique entre plateformes. Repo public (Apache-2.0) : ne jamais y mettre d'URL Linear, de secrets ni de données utilisateur.

## Commands

```bash
cargo build                        # desktop (onnxruntime téléchargé au build, feature ort-download)
cargo test                         # tests unitaires ; SYNAPSE_MODEL_DIR=<dir> active les tests d'embedding
cargo run --example embed_cli -- <model_dir> [texte]   # smoke/parité on-device

# Wheel Python (host desktop) — TOUJOURS via maturin, jamais cargo build direct
cd crates/synapse-core-py && maturin build --release

# Android (l'app embarque libonnxruntime.so, ort le charge dynamiquement)
cargo ndk -t arm64-v8a build -p synapse-core-ffi --no-default-features --features ort-dynamic --release

# Binding Kotlin (UniFFI proc-macro : se génère depuis la lib compilée)
cargo build -p synapse-core-ffi
cargo run -p synapse-core-ffi --bin uniffi-bindgen -- generate \
  --library target/debug/libsynapse_core_ffi.dylib --language kotlin --out-dir bindings/kotlin

# Parité vs Python prod (depuis le venv du backend synapse, wheel installée)
python tools/parity_check.py --model-dir ~/.synapse/models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q
python tools/gen_reference_vectors.py > reference_vectors.json   # golden vectors pour les tests on-device
```

## Module & refacti conventions

Workspace à 3 crates, avec une règle dure : **toute la logique vit dans `crates/synapse-core`, les bindings sont des wrappers sans cervelle** (conversion de types + mapping d'erreurs, rien d'autre). Si un binding a besoin d'un comportement, il se code dans le core.

- `crates/synapse-core` : lib pure Rust. Un module par domaine — `embedder.rs`, `storage.rs`/`schema.rs`/`migrate.rs` (T1), `sql.rs` (passerelle SQL des hôtes), `llm.rs` + `routing.rs` (T2, classif + routing), `sync.rs` (T3, HLC/LWW + `dedup_after_pull` SYN-133), `decay.rs`/`summaries.rs`/`digest.rs`/`resources.rs` (T5), `snapshot.rs` (lecture locale : rend le même JSON que les endpoints de lecture de l'hôte — le SQL y est un miroir délibéré de `api/app.py` côté backend, faire évoluer les deux ensemble ; couvre space+devices, les faits de projet, les files « À valider » tâches/liens et le graphe de la carte SYN-145 — communautés par propagation de labels déterministe, jamais de layout/hulls : caches backend), `actions.rs` (écritures locales SYN-135/139/143/144 : `apply_action` = 33 actions miroir des endpoints d'écriture du backend, même règle de miroir que snapshot.rs ; rejeux caducs = statuts `not_found`/`skipped`, jamais d'erreur bloquante pour la file de l'app). `pairing.rs` couvre aussi le canal code 6 chiffres (SYN-137 : SPAKE2 symétrique + MAC de confirmation `code_confirm_mac/verify` ; un code court dans le HKDF serait brute-forçable hors ligne — jamais ça) ; `seal`/`open` prennent les messages de handshake en slices (clés X25519 32 o ou messages SPAKE2 33 o). `lib.rs` reste un point d'entrée mince qui ré-exporte.
- **Règle transactionnelle** : ce qui doit s'exécuter dans la transaction du caller s'expose sur `SqlConnection` (insert_fact, decay, gather_week, add_project_entry, read_snapshot) ; les passes LLM/vecteurs vivent sur `Brain` (sa propre connexion) et s'appellent hors transaction hôte — sinon SQLITE_BUSY.
- **Prompts** : `prompts/*.md` + `manifest.json`, byte-identiques aux constantes Python historiques (`llm::load_prompt` strippe le `\n` final). Déployés côté hôtes dans `~/.synapse/prompts/` — les livrer avant/avec toute wheel qui les lit.
- **Multilingue (SYN-119, FR/EN-first, extensible)** : les prompts sont **EN-base neutre**, la sortie suit la langue de la capture ; prédicats/types restent EN snake_case (interlingua). Le classifieur émet un champ `"language"` (ISO 639-1) — **détection 100% côté serveur, dans le prompt** (zéro dépendance/appel en plus). Stocké en `atomic_notes.language` (lu du JSON dans `routing.rs::persist_atomic_note`). Les résumés d'entité ne peuvent PAS inférer la langue (faits/relations = prédicats EN + noms propres) → `summaries.rs::dominant_note_language` fait un **vote majoritaire déterministe** sur `atomic_notes.language`, injecté via `{language}` dans `resummary.md`. Digest/resource-summary infèrent bien (vraie prose en entrée). Garde-fou = harness Haiku côté repo `synapse` (`scripts/lang_harness.py`), 0 régression FR. Ajouter une langue = 0 travail cœur.
- `crates/synapse-core-py` : PyO3, module Python `synapse_core` (le crate/fn s'appelle `synapse_core_py` pour éviter la collision de noms avec la dépendance). Exclu des `default-members` : build via **maturin uniquement** (link `-undefined dynamic_lookup`). Libérer le GIL (`py.detach`) autour de tout appel coûteux.
- `crates/synapse-core-ffi` : UniFFI proc-macro (`uniffi::setup_scaffolding!`). Sorties générées dans `bindings/` (gitignoré, régénérable).
- **Les fichiers modèle et les prompts sont de la donnée, jamais du code** : passés en chemin au runtime, bundlés en assets côté apps, jamais commités (App Store 2.5.2). Le comportement (ex. troncature) se lit depuis les fichiers modèle, pas de constantes en dur qui divergeraient de Python.
- Features : `ort-download` (défaut, desktop) vs `ort-dynamic` (mobile, l'app fournit libonnxruntime). Tout nouveau code doit compiler sous les deux.
- Cleanup structurel = commit dédié, jamais mélangé à une feature.

### Pièges verrouillés (ne pas re-payer, cf. SYN-109)

- **Matrice de versions** : fastembed 5.17 → ort `=2.0.0-rc.12` (api-24) → onnxruntime ≥ 1.24 (AAR Android `1.27.0`). Un onnxruntime trop vieux ne donne pas d'erreur : ort rc.12 **deadlocke** (OnceLock réentrant dans son chemin d'erreur). Ne bumper fastembed/ort qu'ensemble, en revérifiant la matrice.
- **`ORT_DYLIB_PATH` ne doit jamais pointer vers un chemin inexistant** (même deadlock). Sur Android (`extractNativeLibs=false`), les .so ne sont pas extraits sur disque : précharger via `System.loadLibrary("onnxruntime")` et laisser ort dlopen par soname.
- Troncature embeddings : le modèle qdrant tronque à **128 tokens** (min de `max_length`/`model_max_length` dans `tokenizer_config.json`, comme fastembed Python) — et 128 est le BON granule pour ce modèle de phrases : embedder un texte long en un seul vecteur 512 **dilue** (mesuré SYN-118 : les requêtes tête ET queue chutent). Depuis SYN-118 les textes longs sont **chunkés** : `Embedder::embed_chunks` = un vecteur par fenêtre de ~128 tokens (overlap 24, max 16 fenêtres) ; notes = une ligne vec0 par chunk (clé `uuid` puis `uuid#k`, `search_notes` déduplique au meilleur chunk) ; ressources = frames concaténées dans le BLOB (`score_against_frames` = max). vec0 ne supporte pas INSERT OR REPLACE : l'upsert balaie puis insère.
- `intra_threads(1)` sur mobile.

## Tests

- Framework : `cargo test` (tests unitaires dans les modules). Les tests nécessitant le modèle lisent `SYNAPSE_MODEL_DIR` et se **skippent** (pas d'échec) s'il est absent : la suite reste exécutable offline et sans les 235 Mo de modèle.
- **Règle stricte : aucune feature ne ship sans test.** Pour tout ce qui touche la parité avec Python (embeddings, et bientôt stockage/routing), le test de référence est un **golden test** : vecteurs/sorties générés côté Python prod (`tools/gen_reference_vectors.py`), comparés côté Rust (seuils : cosinus > 0.9999, cf. `tools/parity_check.py`). C'est le contrat du repo.
- Validation on-device : `examples/embed_cli.rs` (CLI + mode parité + watchdog SIGUSR1) pour Android via adb ; le test instrumenté complet vit dans `synapse-app` (`EmbedderParityTest`).
- Lancer `cargo test` (les deux features si le build system est touché) avant tout commit.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Changelog & doc prod

Suivis dans Linear, hors du repo (repo public : aucune URL Linear dans les fichiers commités ; les identifiants `SYN-123` dans les messages de commit sont OK). Les liens et le rituel de mise à jour sont dans `CLAUDE.local.md` (non commité).

## Environment

Pas de `.env`. Variables utiles :
- `SYNAPSE_MODEL_DIR` : dossier des fichiers modèle (ex. `~/.synapse/models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q`) ; active les tests d'embedding.
- `ORT_DYLIB_PATH` : chemin explicite vers libonnxruntime pour les builds `ort-dynamic` (CLI on-device). Ne jamais le définir vers un chemin inexistant, et ne pas le définir dans une app Android.

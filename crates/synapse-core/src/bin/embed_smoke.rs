//! Embedding parity probe. Prints `{sentence: vector}` as JSON on stdout.
//!
//! CI builds it twice — host macOS and iOS simulator — runs both against the
//! same model directory and diffs the outputs (the core's parity contract).
//! Same role as the Android `embed_cli` harness had for the Pixel.

use std::env;

const SENTENCES: &[&str] = &[
    "Bonjour, ceci est une note de test.",
    "The quick brown fox jumps over the lazy dog.",
    "Réunion demain à 14h avec l'équipe produit pour le point d'étape.",
    "手机上的数据同步引擎运行良好。",
    "Écrire une note avec des accents éàüïœç et des emojis 🚀🧠.",
    "acheter du pain",
    // Over 128 tokens on purpose: exercises the truncation rule the embedder
    // reads from tokenizer_config.json (max_length=128, not 512).
    "La synchronisation pair-à-pair repose sur un journal de versions par \
     colonne, une horloge hybride logique et des tombstones conservées au \
     niveau de la ligne. Chaque appareil applique les changements distants \
     sous un drapeau qui coupe ses propres triggers, puis réinscrit les \
     versions gagnantes dans son journal afin de pouvoir les relayer vers un \
     troisième pair sans jamais produire d'écho. Les curseurs sont locaux à \
     chaque paire d'appareils et avancent page par page pendant le transfert, \
     ce qui rend la reprise après une coupure réseau triviale : on repart du \
     dernier curseur persisté et les lignes complètes garantissent qu'un \
     pair vierge peut se reconstruire entièrement depuis zéro, y compris les \
     lignes ressuscitées après une suppression concurrente.",
];

fn main() {
    let model_dir = env::args()
        .nth(1)
        .expect("usage: embed_smoke <model_dir>");
    let embedder = synapse_core::Embedder::new(&model_dir).expect("embedder init failed");
    let mut out = serde_json::Map::new();
    for s in SENTENCES {
        let v = embedder.embed(s).expect("embed failed");
        out.insert(s.to_string(), serde_json::json!(v));
    }
    println!("{}", serde_json::Value::Object(out));
}

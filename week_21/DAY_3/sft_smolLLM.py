# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:17:28 2026

@author: milos
"""

# ================================================================
# SUPERVISED FINE-TUNING (SFT) — KONZEPTÜBERSICHT
# ================================================================
#
# Supervised Fine-Tuning (SFT) ist der Prozess, bei dem ein vortrainiertes
# Sprachmodell — das Text statistisch vorhersagen kann — darauf trainiert wird,
# sich korrekt zu VERHALTEN: Anweisungen folgen, Fragen beantworten,
# ein Chat-Format einhalten und nützliche Antworten produzieren.

# ── DIE LLM-TRAININGSPIPELINE ────────────────────────────────
#
# SFT ist Schritt 2 eines 3-stufigen Prozesses, der aus rohem Sprachmodelling
# einen einsatzbereiten Assistenten macht:
#
#  1. Pre-Training          2. SFT                    3. RLHF / DPO
#  ─────────────────        ──────────────────────    ──────────────────────
#  Lernt Sprache            Lernt Anweisungen         Lernt menschliche
#  aus ~Billionen           zu folgen, aus            Präferenzen aus
#  ungelabelten Tokens      10k–100k gelabelten       Feedback auf
#                           (Prompt, Antwort)-Paaren  Modellausgaben
#
#  "weiß Dinge"             "weiß, wie man hilft"     "weiß, was sich richtig anfühlt"


# ── WIE SFT MECHANISCH FUNKTIONIERT ─────────────────────────
#
# Während SFT sieht das Modell (Anweisung → ideale Antwort)-Paare
# und aktualisiert seine Gewichte, um den Unterschied zwischen
# seiner Ausgabe und der Zielantwort zu minimieren:
#
#  - Der Verlust (Loss) wird NUR auf den Antwort-Tokens berechnet,
#    nicht auf den Anweisungs-Tokens — das Modell soll nicht für
#    den empfangenen Prompt bestraft werden.
#
#  - Gewichtsaktualisierungen verwenden Standard-Gradientenabstieg mit
#    einer NIEDRIGEREN Lernrate als beim Pre-Training, um bestehendes
#    Wissen nicht zu überschreiben (katastrophales Vergessen).
#
#  - Der "supervised" (überwachte) Aspekt bedeutet: Jedes Beispiel hat
#    eine bekannte korrekte Antwort.


# ── SFT IM VERGLEICH ZU VERWANDTEN TECHNIKEN ────────────────
#
#  Technik         | Benötigte Daten              | Was es lehrt                     | Kosten
#  ────────────────┼──────────────────────────────┼──────────────────────────────────┼────────
#  Pre-Training    | Billionen Tokens, ungelabelt  | Sprache, Fakten, Denken          | Sehr hoch
#  SFT             | 10k–100k gelabelte Paare      | Anweisungsfolgen, Format         | Mittel
#  PEFT / LoRA     | Gleich wie SFT                | Gleich wie SFT, weniger Params   | Niedrig
#  RLHF            | Menschliche Präferenz-Rankings| Feine Ausrichtung, Sicherheit    | Hoch
#  DPO             | Präferenzpaare (gut/schlecht) | Ausrichtung ohne Reward-Modell   | Mittel


# ── WAS SFT TATSÄCHLICH VERÄNDERT ───────────────────────────
#
# SFT injiziert KEIN neues Wissen in das Modell — es verändert,
# WIE das Modell sein bestehendes Wissen anwendet. Deshalb gilt:
#
#  - Ein auf medizinischen Q&A SFT-trainiertes Modell formatiert Antworten
#    wie ein Arzt, kennt aber keine Fakten, die nicht im Pre-Training vorkamen.
#
#  - Domänenspezifisches SFT funktioniert am besten, wenn das Basismodell
#    bereits auf relevanten Daten vortrainiert wurde.
#
#  - SFT kann "katastrophales Vergessen" verursachen — zu aggressives
#    Fine-Tuning auf einem engen Datensatz kann allgemeine Fähigkeiten
#    des Modells verschlechtern.


# In diesem Skript wird SFT auf SmolLM2-135M (ein Basismodell) durchgeführt,
# unter Verwendung von smol-smoltalk (ein Datensatz aus Anweisungs-Antwort-Gesprächen).
#
# Ziel: Ein Modell zu erzeugen, das sich wie sein Instruct-Geschwistermodell
# SmolLM2-135M-Instruct verhält — aber von uns trainiert, auf unserem
# eigenen Datensplit, mit vollständiger Kontrolle über alle Parameter.


from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, clone_chat_template  # setup_chat_format is from trl, not transformers
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. DATASET ──────────────────────────────────────────────
# Use smol-smoltalk (optimized for <1B models, not full smoltalk)
# split= notation returns a Dataset directly (not DatasetDict),
# downloading only 10% of the ~1M rows to save memory/bandwidth
dataset = load_dataset("HuggingFaceTB/smoltalk", "all", split="train[:10%]")

# Create train/eval split (no "test" split exists in this dataset)
split = dataset.train_test_split(test_size=0.05, seed=42)

# ── 2. MODEL & TOKENIZER ────────────────────────────────────
model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# ── 3. CHAT TEMPLATE ────────────────────────────────────────
# adds special tokens, resizes embeddings, syncs eos_token
model, tokenizer, added_tokens = clone_chat_template(
    model,
    tokenizer,
    source_tokenizer_path="HuggingFaceTB/SmolLM2-135M-Instruct"
)

# clone_chat_template only syncs eos_token — pad_token must be set manually
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ── 4. TRAINING CONFIG ──────────────────────────────────────
# Key parameters and their trade-offs:
#   max_length:                  truncates sequences to fit model context window
#   per_device_train_batch_size: larger = more stable gradients, more VRAM
#   gradient_accumulation_steps: simulates larger batches without extra VRAM
#   learning_rate:               5e-5 is a safe starting point for SFT
#   warmup_ratio:                gradual LR ramp-up avoids early instability
#   eval_strategy + eval_steps:  watch for overfitting (val loss rising while
#                                train loss falls)

training_args = SFTConfig(
    output_dir="./sft_output",
    max_steps=1000,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
    max_length =2048,        # ✅ truncate at tokenization time, before hitting the model
)

# ── 5. TRAINER ──────────────────────────────────────────────
# When dataset has a "messages" field, SFTTrainer automatically applies
# the model's chat template — no formatting_func needed
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    processing_class=tokenizer,
)

# Start training
trainer.train()


# Healthy training signs:
#   ✅ Training loss and validation loss both decrease
#   ✅ Small gap between train and val loss (good generalization)
# Warning signs:
#   ⚠️  Val loss rising while train loss falls → overfitting (reduce steps/epochs)
#   ⚠️  No improvement → try higher LR or check data quality
#   ⚠️  Extremely low loss → possible memorization (check output diversity)



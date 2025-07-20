from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import pandas as pd
import random
import re

# === FastAPI uygulaması başlatılıyor
app = FastAPI()

# === Model ve Tokenizer belleğe yükleniyor
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.eval()

# === CSV'den cümleleri al
def load_sentences_from_csv(file_path):
    df = pd.read_csv(file_path)
    all_sentences = []
    for text in df["text"].dropna():
        for sent in text.split('.'):
            sent = sent.strip()
            if len(sent.split()) >= 5:
                all_sentences.append(sent)
    return all_sentences

sentences = load_sentences_from_csv("bbc_text_cls.csv")

# === Tahmin fonksiyonu
def predict_fill_word(prompt_words, blank_index, top_k=15):
    prompt = " ".join(prompt_words[:blank_index])
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, top_k)

    predicted = []
    seen = set()
    for idx in topk.indices:
        word = tokenizer.decode([idx]).strip()
        word = re.sub(r'[^A-Za-z]', '', word).lower()
        if word and word not in seen and len(word) >= 3:
            predicted.append(word)
            seen.add(word)
        if len(predicted) == 4:
            break
    return predicted

# === Rastgele kelimeyi maskele
def mask_random_word(sentence):
    words = sentence.split()
    idx = random.randint(1, len(words) - 2)
    answer = words[idx]
    masked_words = words.copy()
    masked_words[idx] = "____"
    return " ".join(masked_words), answer, idx, words

# === API giriş modeli
class QuizRequest(BaseModel):
    sentence: str = None  # İsteğe bağlı, boşsa sistem cümle seçecek

# === /generate-quiz endpoint'i
@app.post("/generate-quiz")
def generate_quiz(data: QuizRequest):
    if data.sentence:
        sentence = data.sentence
    else:
        sentence = random.choice(sentences)

    masked, answer, idx, words = mask_random_word(sentence)
    options = predict_fill_word(words, idx)
    answer_clean = re.sub(r'[^A-Za-z]', '', answer).lower()

    if answer_clean not in options:
        options.append(answer_clean)
    else:
        options = options[:4] + [answer_clean]

    random.shuffle(options)

    return {
        "masked_sentence": masked,
        "options": options,
        "answer": answer_clean
    }

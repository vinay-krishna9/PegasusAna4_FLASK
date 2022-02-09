import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from flask import Flask, render_template, request, flash

app = Flask(__name__)
app.secret_key = "dwethrbw6443edv6"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/summary", methods=["POST", "GET"])
def summarize():
    modelname = 'google/pegasus-xsum'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(modelname)
    model = PegasusForConditionalGeneration.from_pretrained(
        modelname).to(torch_device)
    batch = tokenizer.prepare_seq2seq_batch(str(
        request.form['data_input']), truncation=True, padding="longest", return_tensors='pt').to(torch_device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    flash(tgt_text[0])
    return render_template("index.html")

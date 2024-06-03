from pycocoevalcap.spice.spice import Spice
import evaluate

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

def compute_metrics(eval_pred, tokenizer):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids

    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge_result = rouge.compute(predictions=pred_str, references=labels_str)
    rouge_result = {k: round(v * 100, 4) for k, v in rouge_result.items()}

    bleu_result = bleu.compute(predictions=pred_str, references=labels_str)
    bleu_score = round(bleu_result["bleu"] * 100, 4)

    spice_result = compute_spice(pred_str, labels_str)
    spice_score = round(spice_result["spice"], 4)

    return {
        "rouge1": rouge_result.get("rouge1", 0),
        "rouge2": rouge_result.get("rouge2", 0),
        "rougeL": rouge_result.get("rougeL", 0),
        "bleu": bleu_score,
        "spice": spice_score
    }

def compute_spice(predictions, references):
    spice = Spice()
    res = {i: [pred] for i, pred in enumerate(predictions)}
    gts = {i: [ref] for i, ref in enumerate(references)}
    average_score, _ = spice.compute_score(gts, res)
    return {"spice": average_score}

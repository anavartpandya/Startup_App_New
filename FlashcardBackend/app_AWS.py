import json
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load the fine-tuned model weights
model_path = 'path/to/your/fine_tuned_bert_sst2_2.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

def predict_sentiment(input_text):
    model.eval()
    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        input_text,
        None,
        add_special_tokens=True,
        max_length=512,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()  # Get the predicted class
    return prediction

def lambda_handler(event, context):
    # Parse the input data
    try:
        data = json.loads(event['body'])
        front_text = data['frontText']
        back_text = data['backText']

        # Predict sentiment
        front_text_pred = 'positive' if predict_sentiment(front_text) == 1 else 'negative'
        back_text_pred = 'positive' if predict_sentiment(back_text) == 1 else 'negative'

        # Return the response
        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'frontText': front_text,
                'frontTextPred': front_text_pred,
                'backText': back_text,
                'backTextPred': back_text_pred
            })
        }
        return response

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import spacy

# Load spaCy NER model for entity detection
nlp = spacy.load("en_core_web_sm")

# Load GENRE model and tokenizer
model_name = "facebook/genre-linking-aidayago2"  # You can use any available GENRE model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Full text to process
text = """Barack Obama, the 44th President of the United States, was born in Honolulu, Hawaii. 
He served two terms in office from 2009 to 2017. 
During his presidency, Obama signed the Affordable Care Act into law, 
a significant reform of the healthcare system. In 2015, Obama visited Cuba, 
marking the first time a sitting U.S. president had visited the island in nearly 90 years. 
After leaving office, he continued his work on global issues and humanitarian causes."""

# Process raw text with spaCy NER to get entities
doc = nlp(text)
entities = [ent.text for ent in doc.ents]

# Perform entity linking with GENRE for each detected entity
for entity in entities:
    # Link entity using GENRE (mGENRE will generate a valid entity name from input text)
    inputs = tokenizer([f"[START] {entity} [END]"], return_tensors="pt")  # You can skip [START]/[END] for GENRE
    outputs = model.generate(**inputs)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Print the result
    print(f"Entity: {entity} → Linked to: {result}")

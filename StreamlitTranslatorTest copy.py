# Streamlit implementation of South American Translator tool
# Nabeel Paruk
# Creasted: 30/08/2024 13:50:00
# Last Updated: 30/08/2024 13:50:00

# Import modules
import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import random
from keras import ops
import pickle
import keras_nlp

# Display working directory
cwd = os.getcwd()
print(cwd)

# Define parameters and hyperparameters
MAX_SEQUENCE_LENGTH = 40

# Functions
def load_tokenizer(lan_set, lan):
    """
    Loads a saved tokenizer
    Args:
        lan_set: Language set which the tokenizer belongs to e.g. engpor
        lan: Which language in the language set you want e.g. por
    """
    if lan_set == "engspa":
        folder = "Spanish"
    elif lan_set == "engpor":
        folder = "Portuguese"

    with open(f"/Users/nabeel.paruk/Documents/My stuff/SouthAmericanTranslator/Tokenizers/English{folder}/tokenizer_{lan_set}_{lan}.pickle", 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def decode_sequences(input_sentences, transformer, lan_tokenizer, eng_tokenizer):
    """
    Intermediate step in the translating process. Decodes the encoded sample by taking the character with the highest predicted probability.
    Args:
        input_sentences: Sentence to translate
        transformer: Transformer that works with language that you want to translate (Eng-Spa - transformer1, Eng-Por = transformer2)
        lan_tokenizer: The tokenizer for the language you want to translate from. This must work with your transformer e.g. if transformer2 use por_tokenizer
    """
    batch_size = 1
    
    # Tokenize encoder input
    encoder_input_tokens = ops.convert_to_tensor(eng_tokenizer(input_sentences))
    if len(encoder_input_tokens[0]) < MAX_SEQUENCE_LENGTH:
        pads = ops.full((1, MAX_SEQUENCE_LENGTH - len(encoder_input_tokens[0])), 0)
        encoder_input_tokens = ops.concatenate(
            [encoder_input_tokens.to_tensor(), pads], 1
        )
        
    # Define a function that outputs the next tokens probability given the input sequence
    def next(prompt, cache, index):
        logits = transformer([encoder_input_tokens, prompt])[:, index - 1, :]
        # We ignore hidden states for now -> needed only for contrastive search
        hidden_states = None
        return logits, hidden_states, cache
    
    # Build a prompt of length 40 with a start token and padding tokens
    length = 40
    # Add start token
    start = ops.full((batch_size, 1), lan_tokenizer.token_to_id("[START]"))
    # Add pad token
    pad = ops.full((batch_size, length - 1), lan_tokenizer.token_to_id("[PAD]"))
    
    prompt = ops.concatenate((start, pad), axis=-1)
    
    # GreedySampler -> Outputs token with highest probability
    generated_tokens = keras_nlp.samplers.GreedySampler()(
        next,
        prompt,
        stop_token_ids=[lan_tokenizer.token_to_id("[END]")],
        index=1, # Sample only after "[START]" token
    )
    generated_sentences = lan_tokenizer.detokenize(generated_tokens)
    return generated_sentences

def translate(text_to_translate, transformer, lan_tokenizer, eng_tokenizer):
    """
    Takes an input sentence and translates it
    Args:
        text_to_translate: Input sentence
        transformer: Transformer to use to translate depending on languages e.g. if Portuguese use transformer2
        lan_tokenizer: The tokenizer for the language you want to translate from. This must work with your transformer e.g. if transformer2 use tokenizer_engpor_por
        eng_tokenizer: The English associated tokenizer with the language you want to translate from e.g. tokenizer_engpor_eng
    """
    input_sentence = text_to_translate
    translated = decode_sequences([input_sentence], transformer, lan_tokenizer, eng_tokenizer)
    translated = translated.numpy()[0].decode('utf-8')
    
    translated = (
        translated.replace("[PAD]", "")
        .replace("[UNK]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .strip()
    )
    return translated

def main():

    # Load models
    transformer1 = tf.keras.models.load_model("/Users/nabeel.paruk/Documents/My stuff/SouthAmericanTranslator/Models/transformer_eng_spa.keras")
    transformer2 = tf.keras.models.load_model("/Users/nabeel.paruk/Documents/My stuff/SouthAmericanTranslator/Models/transformer_eng_spa.keras")

    # Load tokenizers
    # - Spanish
    tokenizer_engspa_eng = load_tokenizer("engspa", "eng")
    tokenizer_engspa_spa = load_tokenizer("engspa", "spa")

    # - Portuguese
    tokenizer_engpor_eng =load_tokenizer("engpor", "eng")
    tokenizer_engpor_por = load_tokenizer("engpor", "por")

    # Interface
    # - Header and Description
    st.header("South American Translator Tool")
    st.write("*More languages coming soon!*")

    # - Choosing the language
    chosen_language = st.radio(
        "Language to translate to:",
        ["Spanish", "Portuguese"]
        )
    
    if chosen_language == "Spanish":
        transformer = transformer1
        lan_tokenizer = tokenizer_engspa_spa
        eng_tokenizer = tokenizer_engspa_eng

    elif chosen_language == "Portuguese":
        transformer = transformer2
        lan_tokenizer = tokenizer_engpor_por
        eng_tokenizer = tokenizer_engpor_eng
        

    # - Allow text input
    text_to_translate = st.text_input("Enter your English sentence")

    # - Translate and output
    translation = translate(text_to_translate, transformer, lan_tokenizer, eng_tokenizer)
    if text_to_translate:
        st.write(translation)


if __name__ == '__main__':
    main()
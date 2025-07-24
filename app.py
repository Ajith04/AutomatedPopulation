import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io

st.set_page_config(page_title="Excel Mapper", layout="centered")
st.title("📊 AI-Based Excel Mapper")

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Load reference entity rows as full sentences
def load_entity_sentences(df):
    return [' '.join([f"{col} is {str(val)}" for col, val in row.items()]) for _, row in df.iterrows()]

# Find best match for a client row
def find_best_match(sentence, ref_sentences, ref_df):
    client_emb = model.encode([sentence])
    ref_emb = model.encode(ref_sentences)
    sims = cosine_similarity(client_emb, ref_emb)[0]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] > 0.7:
        return ref_df.iloc[best_idx]
    return None

# Main mapping function
def generate_output(client_df, reference_sheets, template_df):
    filled_rows = []
    for _, client_row in client_df.iterrows():
        row_sentence = ' '.join([str(val) for val in client_row.values])
        best_match = None

        for sheet_name, ref_df in reference_sheets.items():
            ref_sentences = load_entity_sentences(ref_df)
            match = find_best_match(row_sentence, ref_sentences, ref_df)
            if match is not None:
                best_match = match
                break

        if best_match is not None:
            filled_rows.append(best_match.to_dict())
        else:
            filled_rows.append({col: None for col in template_df.columns})

    return pd.DataFrame(filled_rows)

# Uploads
client_file = st.file_uploader("📁 Upload Client Data Excel", type=["xlsx"])
reference_file = st.file_uploader("📁 Upload System Instructions Excel", type=["xlsx"])

if client_file and reference_file:
    st.success("Files uploaded! Ready to process.")
    if st.button("🟢 Generate Finalized Template"):
        client_df = pd.read_excel(client_file)
        ref_xl = pd.ExcelFile(reference_file)
        reference_sheets = {sheet: ref_xl.parse(sheet) for sheet in ref_xl.sheet_names}

        # Use first sheet of instructions as the template structure
        template_df = list(reference_sheets.values())[0].iloc[0:0].copy()

        result_df = generate_output(client_df, reference_sheets, template_df)

        # Output download
        output = io.BytesIO()
        result_df.to_excel(output, index=False)
        st.download_button("📥 Download Final Output", output.getvalue(), "final_template.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

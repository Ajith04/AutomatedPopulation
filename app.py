import streamlit as st
import pandas as pd
import torch
import os
from sentence_transformers import SentenceTransformer, util

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="Excel Data Mapper", layout="centered")
st.title("🧠 Intelligent Excel Data Mapper")
st.markdown("""
This tool semantically maps client data values to your system's expected values using AI.

- **Client Data Sheet**: Rows of real venue/facility records (e.g., Name, VenueType, etc.)
- **Reference Sheet**: Allowed values per attribute, column-wise
- **Output**: Populated template with AI-assisted mappings
""")

# Upload files
client_file = st.file_uploader("📥 Upload Client Data Excel", type=["xlsx"])
reference_file = st.file_uploader("📄 Upload System Reference Excel", type=["xlsx"])

# Utility: cosine similarity with safety
def safe_cos_sim(a, b):
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if b.shape[0] == 0:  # No reference embeddings
        return torch.tensor([[0.0]])
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

# Main logic
if client_file and reference_file:
    client_df = pd.read_excel(client_file)
    ref_df = pd.read_excel(reference_file)
    result_df = pd.DataFrame(columns=client_df.columns)

    # Prepare reference embeddings
    reference_embeddings = {}
    for col in ref_df.columns:
        ref_values = ref_df[col].dropna().astype(str).unique().tolist()
        if ref_values:
            embeddings = model.encode(ref_values, convert_to_tensor=True)
        else:
            embeddings = torch.empty((0, model.get_sentence_embedding_dimension()))
        reference_embeddings[col] = {"values": ref_values, "embeddings": embeddings}

    # Process each row
    for _, row in client_df.iterrows():
        mapped_row = {}
        for col in client_df.columns:
            val = str(row[col]).strip()
            ref = reference_embeddings.get(col)

            # Exact match
            if ref and val in ref["values"]:
                mapped_row[col] = val

            # Try semantic match
            elif ref:
                if val == "" or len(ref["values"]) == 0:
                    mapped_row[col] = val
                else:
                    val_embedding = model.encode(val, convert_to_tensor=True)
                    sim_scores = safe_cos_sim(val_embedding, ref["embeddings"])
                    best_idx = sim_scores[0].argmax().item()
                    best_score = sim_scores[0][best_idx].item()

                    if best_score > 0.5:
                        mapped_row[col] = ref["values"][best_idx]
                    else:
                        mapped_row[col] = val  # fallback to original
            else:
                mapped_row[col] = val  # no reference for this column

        result_df.loc[len(result_df)] = mapped_row

    # Final Output
    st.success("✅ Mapping complete! Download the final output.")
    output_file = "mapped_output.xlsx"
    result_df.to_excel(output_file, index=False)

    with open(output_file, "rb") as f:
        st.download_button("📥 Download Mapped Excel", f, file_name="mapped_output.xlsx")

    os.remove(output_file)


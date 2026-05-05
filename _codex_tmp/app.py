import json
from pathlib import Path

import streamlit as st

from llm.model import DEFAULT_MODEL_PATH, ask_llm, load_model
from processing.structurer import build_pid_document, retrieve_relevant_chunks


st.set_page_config(page_title="P&ID QA System", layout="wide")
st.title("P&ID QA System (Phi-3 Local)")
st.caption(
    "Upload a digital P&ID PDF. The app extracts the text, creates structured "
    "JSON in the backend, and answers questions from that generated data."
)


@st.cache_data(show_spinner=False)
def process_pdf(pdf_bytes: bytes, filename: str) -> dict:
    return build_pid_document(pdf_bytes, filename)


def build_document_preview(document: dict, max_pages: int = 3, max_chunks: int = 5) -> dict:
    return {
        "metadata": document["metadata"],
        "pages": document["pages"][:max_pages],
        "chunks": document["chunks"][:max_chunks],
    }


with st.sidebar:
    st.subheader("Model Settings")
    model_path = st.text_input("GGUF model path", value=DEFAULT_MODEL_PATH)
    top_k = st.slider("Relevant chunks", min_value=2, max_value=8, value=4)
    max_tokens = st.slider("Answer length", min_value=128, max_value=512, value=256, step=32)

uploaded_file = st.file_uploader("Upload a digital P&ID PDF", type=["pdf"])
question = st.text_input("Ask a question about the P&ID")

document = None
if uploaded_file is not None:
    try:
        with st.spinner("Extracting text and generating backend JSON..."):
            document = process_pdf(uploaded_file.getvalue(), uploaded_file.name)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Could not process the uploaded PDF: {exc}")
        st.stop()

    metadata = document["metadata"]
    st.info(
        f"Backend JSON created for {metadata['page_count']} page(s) with "
        f"{metadata['chunk_count']} chunk(s) and {metadata['extracted_char_count']} "
        "extracted characters."
    )

    if metadata["extracted_char_count"] == 0:
        st.warning(
            "No selectable text was extracted. This flow expects a digital PDF, "
            "not a scanned image."
        )
    else:
        json_payload = json.dumps(document, indent=2)
        preview_col, download_col = st.columns([3, 1])

        with preview_col:
            with st.expander("Backend JSON Preview"):
                st.json(build_document_preview(document))

        with download_col:
            st.download_button(
                "Download JSON",
                data=json_payload,
                file_name=f"{Path(uploaded_file.name).stem}.json",
                mime="application/json",
                use_container_width=True,
            )

if st.button("Get Answer", type="primary"):
    if uploaded_file is None:
        st.warning("Upload a digital P&ID PDF first.")
    elif not question.strip():
        st.warning("Enter a question to continue.")
    elif document is None or not document["chunks"]:
        st.warning("No extracted document context is available yet.")
    else:
        retrieved_chunks = retrieve_relevant_chunks(document, question, top_k=top_k)

        if not retrieved_chunks:
            st.warning("No relevant extracted text was found for this question.")
        else:
            with st.expander("Retrieved Context", expanded=True):
                for chunk in retrieved_chunks:
                    st.markdown(
                        f"**Page {chunk['page_number']} | Chunk {chunk['chunk_number']}**"
                    )
                    st.code(chunk["text"], language="text")

            try:
                with st.spinner("Loading local model..."):
                    llm = load_model(model_path=model_path)
                with st.spinner("Generating answer..."):
                    answer = ask_llm(
                        llm=llm,
                        question=question,
                        retrieved_chunks=retrieved_chunks,
                        metadata=document["metadata"],
                        max_tokens=max_tokens,
                    )
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Model inference failed: {exc}")
            else:
                if answer:
                    st.success(answer)
                else:
                    st.warning("The model did not return an answer.")

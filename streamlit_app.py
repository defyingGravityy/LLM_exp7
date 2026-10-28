import os
import time
import streamlit as st

# On Streamlit Cloud, values in the Secrets panel become environment variables.
# This also lets support_rag.py read them via os.getenv. Set BEFORE importing it.
try:
    for k, v in st.secrets.items():
        if isinstance(v, str):
            os.environ.setdefault(k, v)
except Exception:
    pass  # no secrets file locally; .env is used instead

import support_rag
from support_rag import BackendError

st.set_page_config(page_title="Customer Support RAG", page_icon="💬")
st.title("💬 Customer Support Assistant")
st.caption("Answers come only from the FAQ knowledge base.")


@st.cache_resource
def warm_up():
    # support_rag builds embeddings + FAISS at import; cache_resource keeps it alive across reruns
    return support_rag


warm_up()

question = st.text_input("Ask a question", max_chars=1000, placeholder="How do I request a refund?")

if st.button("Ask") or question:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        start = time.perf_counter()
        try:
            with st.spinner("Thinking..."):
                result = support_rag.answer_question(question.strip())
        except BackendError:
            st.error("The service is temporarily unavailable. Please try again later.")
        except Exception:
            st.error("Something went wrong. Please try again.")
        else:
            ms = (time.perf_counter() - start) * 1000
            st.write(result["answer"])
            if result["grounded"]:
                st.caption("Sources: " + ", ".join(result["sources"]))
            else:
                st.info("No reliable answer found in the FAQ.")
            st.caption(f"Intent: {result['intent']} · {ms:.0f} ms")

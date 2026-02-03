from logger_config import  logger
import cv2
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_qdrant import QdrantVectorStore
from ultralytics import YOLO
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Constants
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bouye_laws"
MODEL_PATH = "../models/best.pt"

# Initialize Vision Model
vision_model = YOLO(MODEL_PATH)
llm = OllamaLLM(model="llama3.2:1b", temperature=0.1)

# Define a professional prompt for the Skipper
prompt = ChatPromptTemplate.from_template("""
As a professional Maritime Skipper, provide a short and clear instruction.
Detected object: {object}
Official IALA Rules: {rules}

Provide a concise command for the crew. If no object, say it's clear.
Command:
""")

class SkipperState(TypedDict):
    image_path: str
    detected_objects: List[str]
    navigation_rules: str
    decision: str


def get_vector_store() -> Optional[QdrantVectorStore]:
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            url=QDRANT_URL
        )
    except Exception as e:
        logger.error(f"Vector store connection failed: {e}")
        return None


# Global vector store instance
vector_store = get_vector_store()


def perception_node(state: SkipperState):
    logger.info(f"Processing image: {state['image_path']}")

    results = vision_model.predict(source=state['image_path'], conf=0.5, verbose=False)
    found_classes = []

    for r in results:
        for box in r.boxes:
            label = vision_model.names[int(box.cls[0])]
            found_classes.append(label)

    logger.info(f"Vision results: {found_classes}")
    return {"detected_objects": found_classes}


def knowledge_node(state: SkipperState):
    detected = state.get('detected_objects', [])

    if not detected:
        return {"navigation_rules": "No objects detected in sight."}

    if not vector_store:
        logger.warning("Vector store not initialized, skipping RAG.")
        return {"navigation_rules": "Database connection error."}

    # Querying the first detected object
    query = f"Navigation rules for: {detected[-1]}"
    try:
        results = vector_store.similarity_search(query, k=1)
        rule_text = results[0].page_content if results else "Rule not found in manual."
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        rule_text = "Error retrieving rules."

    return {"navigation_rules": rule_text}


def navigation_node(state: SkipperState):
    logger.info("LLM is reasoning about the situation...")

    detected = state.get("detected_objects", [])
    rules = state.get("navigation_rules", "")

    if not detected:
        decision = "KEEP COURSE: All clear."
    else:
        # LLM creates a professional response based on RAG data
        chain = prompt | llm
        try:
            decision = chain.invoke({
                "object": detected[0],
                "rules": rules
            })
        except Exception as e:
            logger.error(f"LLM Reasoning failed: {e}")
            decision = f"CAUTION: {detected[0]} detected. Follow IALA rules."

    logger.info(f"Final Decision: {decision}")
    return {"decision": decision}


# Graph Construction
workflow = StateGraph(SkipperState)

workflow.add_node("perception", perception_node)
workflow.add_node("knowledge", knowledge_node)
workflow.add_node("navigation", navigation_node)

workflow.set_entry_point("perception")
workflow.add_edge("perception", "knowledge")
workflow.add_edge("knowledge", "navigation")
workflow.add_edge("navigation", END)

app = workflow.compile()

# Example usage
if __name__ == "__main__":
    # 1. Define the input
    sample_input = {"image_path": "../test_buoy.jpg"}

    # 2. Fire up the brain
    logger.info("Starting AI Skipper inference...")
    result = app.invoke(sample_input)

    # 3. Print the final result
    print("\n" + "=" * 50)
    print(f"FINAL SYSTEM DECISION: {result['decision']}")
    print("=" * 50)
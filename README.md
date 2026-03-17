# MindRAG
Agent-based Condition Monitoring Assistance with Multimodal Industrial Database Retrieval Augmented Generation
![Abstract fig brief(1)](https://github.com/user-attachments/assets/86bd901f-3471-4d84-88e1-4dd5f4b4da11)

The MindRAG framework consists of LLM agents connected to a process industry database with a set of custom tools, which allows for signal-to-text analysis of unlabelled, annotated industrial condition monitoring data.

The responses are grounded in retrieved data and embedded knowledge through prompts and readable files, which limits the risk for hallucinations and incorrect generalisations. The quality of the response is thus directly tied to the multimodal retrieval mechanism, which integrates signal data, text data, and machine hierarchy data, to search for the most similar cases in these domains.

The agent can be called with:

Call signal-to-text with:
```
await SIGRAG_test_functions.test_SIGRAG_agent(vector_store = vector_store, source_vector_store = source_vector_store, 
                                              filtered_vector_store = filtered_vector_store, 
                                              prediction_agent = prediction_agent, evaluation_agent = evaluation_agent)
```

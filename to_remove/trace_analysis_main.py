
trace_identifiers = {
    "definition": "<definition>",
    "bloom": "<bloom>",
    "reconstruction": "<reconstruction>", 
    "final_answer": "<final>",
}

tagged_trace_path = 'trace_analysis_outputs/trace_analysis_results_2025-09-15_22-58-21.json'
annotated_traces = load_annotated_traces(annotated_trace_path)

annotated_traces = analyze_trace_identifiers(annotated_traces, trace_identifiers, tokenizer)

print_trace_analysis(annotated_traces)

from graph.workflow import summarizer


def main() -> None:
    print("\nEnter your paragraph.")
    print("Press ENTER on an empty line when finished.\n")

    paragraph_lines: list[str] = []
    while True:
        line = input()
        if not line.strip():
            break
        paragraph_lines.append(line)

    paragraph = "\n".join(paragraph_lines).strip()
    if not paragraph:
        print("Error: Please enter a paragraph.")
        return

    result = summarizer.invoke(
        {"paragraph": paragraph, "summary": "", "summary_length": "Medium"}
    )
    print("\n========== SUMMARY ==========")
    print(result["summary"])
    print("=============================\n")
    if result.get("memory_storage_warning"):
        print(f"Memory storage warning: {result['memory_storage_warning']}")
    elif result.get("memory_storage_count"):
        print(f"Saved {result['memory_storage_count']} extracted memories to PostgreSQL.")


if __name__ == "__main__":
    main()

import generate_graph as gg
import process_text

if __name__ == "__main__":
    #result = gg.get_graph()
    process_text.process("train")
    process_text.process("dev_unlabeled")
    process_text.process("validation")

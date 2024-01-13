import cnn
import generate_graph as gg
import process_text
#import fasttext_method as ft
import svm

if __name__ == "__main__":
    #result = gg.get_graph()
    process_text.process("train")
    process_text.process("dev_unlabeled")
    process_text.process("validation")

    #svm.svm_train(file="ft_train", C_value=10, gamma_value='auto', kernel_value='rbf')
    #cnn.cnn_train(file="ft_train", epochs=5)
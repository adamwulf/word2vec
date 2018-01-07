SCRIPTS_DIR=../scripts
CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = -lm -pthread -march=native -Wall  -Wno-unused-result -funroll-loops -Ofast
BIN_DIR=../bin

all: word2gram word2vec word2phrase distance word-analogy compute-accuracy compute-accuracy-syntax cleano

vocab.o : vocab.c vocab.h
	$(CC) -c vocab.c $(CFLAGS)
ngram_tools.o : ngram_tools.c ngram_tools.h
	$(CC) -c ngram_tools.c $(CFLAGS)
trainingThread.o : trainingThread.c trainingThread.h
	$(CC) -c trainingThread.c $(CFLAGS)
word2gram : word2gram.c vocab.o ngram_tools.o trainingThread.o
	$(CC) word2gram.c vocab.o ngram_tools.o trainingThread.o -o ${BIN_DIR}/word2gram $(CFLAGS)
word2vec : word2vec.c word2vec-main.c vocab.o ngram_tools.o trainingThread.o
	$(CC) word2vec.c word2vec-main.c vocab.o ngram_tools.o trainingThread.o -o ${BIN_DIR}/word2vec $(CFLAGS)
word2phrase : word2phrase.c
	$(CC) word2phrase.c -o ${BIN_DIR}/word2phrase $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o ${BIN_DIR}/distance $(CFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o ${BIN_DIR}/word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o ${BIN_DIR}/compute-accuracy $(CFLAGS)
compute-accuracy-syntax : compute-accuracy-syntax.c
	$(CC) compute-accuracy-syntax.c -o ${BIN_DIR}/compute-accuracy-syntax $(CFLAGS)
	chmod +x ${SCRIPTS_DIR}/*.sh

cleano: word2vec word2gram
	rm *.o

clean:
	pushd ${BIN_DIR} && rm -rf word2vec word2phrase distance word-analogy compute-accuracy; popd

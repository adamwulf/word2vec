#ifndef NGRAMTOOLS
#define NGRAMTOOLS
#define DEBUG_MODE 2

typedef float real;

void gramVocToWordVec(vocabulary* voc, real* syn0,int max_string, int layer1_size, int ngram, int hashbang,int group_vec, int binary,int position, char* train_file, char* output_file);
void writeGrams(vocabulary* voc,real *syn0,int layer1_size,int ngram,int hashbang,int position,char* output_file, int binary);
void sumGram(real* syn0, int layer1_size, int offset, real* vector);
void sumFreqGram(real* syn0, int layer1_size,int offset, real* vector,int cn);
void minmaxGram(real* syn0, int layer1_size,int offset,real *vector,int min);
void truncGram(real* syn0, int layer1_size,int ngram,int offset, real *vector, int wordLength, int gramPos);

#endif
//
//  word2vec.h
//  word2vec
//
//  Created by Adam Wulf on 1/7/18.
//

#ifndef word2vec_h
#define word2vec_h

#define MAX_STRING 1000
#define MAX_EXP 6

extern int EXP_TABLE_SIZE;

extern long long file_size, classes;

extern char train_file[MAX_STRING], output_file[MAX_STRING];
extern char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];


extern int binary, cbow, debug_mode, window, min_count, num_threads;
extern long long layer1_size;

typedef float real;     // Precision of float numbers

extern real alpha, starting_alpha, sample;

extern real *syn0, *syn1, *syn1neg, *expTable;
extern int hs, negative;

void TrainModel(vocabulary* voc);
void DestroyNet();

#endif /* word2vec_h */

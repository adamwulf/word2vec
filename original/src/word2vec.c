//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.


/**
* For more details see:
*
* http://arxiv.org/abs/1402.3722
* http://yinwenpeng.wordpress.com/2013/12/18/word2vec-gradient-calculation/
* http://yinwenpeng.wordpress.com/2013/09/26/hierarchical-softmax-in-neural-network-language-model/
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#include "vocab.h"
#include "trainingThread.h"
#include "word2vec-inc.h"

int EXP_TABLE_SIZE = 1000;

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];

int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1;
long long layer1_size = 100;

long long word_count_actual = 0, file_size = 0, classes = 0;

real alpha = 0.025, starting_alpha, sample = 0;
//syn0 = vectors table
real *syn0, *syn1, *syn1neg, *expTable;

clock_t start;

int hs = 1, negative = 0;

const int table_size = 1e8;

int *table;

void InitUnigramTable(vocabulary * voc) {
	int a, i;
	long long train_words_pow = 0;
	real d1, power = 0.75;
	table = (int *)malloc(table_size * sizeof(int));


	for (a = 0; a < voc->vocab_size; a++)
		train_words_pow += pow(voc->vocab[a].cn, power); //occurences^power

	i = 0;
	d1 = pow(voc->vocab[i].cn, power) / (real)train_words_pow; //normalize

	for (a = 0; a < table_size; a++) {

		table[a] = i;

		if (a / (real)table_size > d1) {
		  i++;
		  d1 += pow(voc->vocab[i].cn, power) / (real)train_words_pow;
		}

		if (i >= voc->vocab_size)
			i = voc->vocab_size - 1;
		}
}

void DestroyNet() {
  if (syn0 != NULL) {
    free(syn0);
  }
  if (syn1 != NULL) {
    free(syn1);
  }
  if (syn1neg != NULL) {
    free(syn1neg);
  }
}

void InitNet(vocabulary * voc) {
	long long a, b;
	a = posix_memalign((void **)&syn0, 128, (long long)voc->vocab_size * layer1_size * sizeof(real));

	if (syn0 == NULL) {
		printf("Memory allocation failed\n"); 
		exit(1);
	}

	if (hs) {
		a = posix_memalign((void **)&syn1, 128, (long long)voc->vocab_size * layer1_size * sizeof(real));

		if (syn1 == NULL) {
			printf("Memory allocation failed\n");
			exit(1);
		}

		for (b = 0; b < layer1_size; b++)
			for (a = 0; a < voc->vocab_size; a++)
				 syn1[a * layer1_size + b] = 0;
	}

	if (negative>0) {
		a = posix_memalign((void **)&syn1neg, 128, (long long)voc->vocab_size * layer1_size * sizeof(real));

		if (syn1neg == NULL){
			printf("Memory allocation failed\n");
			exit(1);
		}

		for (b = 0; b < layer1_size; b++)
			for (a = 0; a < voc->vocab_size; a++)
		 		syn1neg[a * layer1_size + b] = 0;
	}

	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < voc->vocab_size; a++)
			syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;

	CreateBinaryTree(voc);
}

void TrainModel(vocabulary* voc) {
	long a, b, c, d;
	FILE *fo;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

	starting_alpha = alpha;
	InitNet(voc);

	if (negative > 0)
		InitUnigramTable(voc);

	start = clock();

	threadParameters* params; 

	for (a = 0; a < num_threads; a++){
		params = CreateParametersStruct(
			voc,
			syn0,
			syn1,
			syn1neg,
			expTable,
			(&alpha),
			starting_alpha,
			sample,
			(&word_count_actual),
			table,
			a,
			num_threads,
			file_size,
			MAX_STRING,
			EXP_TABLE_SIZE,
			0,
			layer1_size,
			window,
			MAX_EXP,
			hs,
			negative,
			table_size,
			0,
			0,
			0,
			train_file
			);

		/*NB: The parameters struct are freed by each thread.*/

		if(cbow)
			pthread_create(&pt[a], NULL, TrainCBOWModelThread, (void *)params);
		else
			pthread_create(&pt[a], NULL, TrainSKIPModelThread, (void *)params);
	}

	for (a = 0; a < num_threads; a++)
		pthread_join(pt[a], NULL);

	if(debug_mode > 0)
		printf("Training Ended !\n");

	fo = fopen(output_file, "wb");


	if (classes == 0) {
		// Save the word vectors
		fprintf(fo, "%lld %d\n", voc->vocab_size, layer1_size);
		for (a = 0; a < voc->vocab_size; a++) {
			fprintf(fo, "%s ", voc->vocab[a].word);

			if (binary)
				for (b = 0; b < layer1_size; b++)
					fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
			else
				for (b = 0; b < layer1_size; b++)
					fprintf(fo, "%lf ", syn0[a * layer1_size + b]);

			fprintf(fo, "\n");
		}
	} else {
		// Run K-means on the word vectors
		int clcn = classes, iter = 10, closeid;
		int *centcn = (int *)malloc(classes * sizeof(int));
		int *cl = (int *)calloc(voc->vocab_size, sizeof(int));
		real closev, x;
		real *cent = (real *)calloc(classes * layer1_size, sizeof(real));

		for (a = 0; a < voc->vocab_size; a++)
			cl[a] = a % clcn;

		for (a = 0; a < iter; a++) {
			for (b = 0; b < clcn * layer1_size; b++)
				cent[b] = 0;

			for (b = 0; b < clcn; b++)
				centcn[b] = 1;

			for (c = 0; c < voc->vocab_size; c++) {

				for (d = 0; d < layer1_size; d++)
					cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];

				centcn[cl[c]]++;
			}

			for (b = 0; b < clcn; b++) {
				closev = 0;

				for (c = 0; c < layer1_size; c++) {
					cent[layer1_size * b + c] /= centcn[b];
					closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
				}

				closev = sqrt(closev);
				for (c = 0; c < layer1_size; c++)
					cent[layer1_size * b + c] /= closev;
			}

			for (c = 0; c < voc->vocab_size; c++) {
				closev = -10;
				closeid = 0;
				for (d = 0; d < clcn; d++) {
					x = 0;
					for (b = 0; b < layer1_size; b++)
						x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];

					if (x > closev) {
						closev = x;
						closeid = d;
					}
				}
				cl[c] = closeid;
			}
		}
		// Save the K-means classes

		for (a = 0; a < voc->vocab_size; a++){
			fprintf(fo, "%s %d ", voc->vocab[a].word, cl[a]);

			for (b = 0; b < layer1_size; b++){
				fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
			}
			fprintf(fo, "\n");
		}

		free(centcn);
		free(cent);
		free(cl);
		
	}

	fclose(fo);
	free(pt);

}




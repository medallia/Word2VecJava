package com.medallia.word2vec.neuralnetwork;

import com.google.common.collect.Multiset;
import com.medallia.word2vec.Word2VecTrainerBuilder.TrainingProgressListener;
import com.medallia.word2vec.huffman.HuffmanCoding.HuffmanNode;

import java.util.List;
import java.util.Map;

/**
 * Trainer for neural network using continuous bag of words
 */
class CBOWModelTrainer extends NeuralNetworkTrainer {
	
	CBOWModelTrainer(NeuralNetworkConfig config, Multiset<String> counts, Map<String, HuffmanNode> huffmanNodes, TrainingProgressListener listener) {
		super(config, counts, huffmanNodes, listener);
	}
	
	/** {@link Worker} for {@link CBOWModelTrainer} */
	private class CBOWWorker extends Worker {
		private CBOWWorker(int randomSeed, int iter, Iterable<List<String>> batch) {
			super(randomSeed, iter, batch);
		}
		
		@Override void trainSentence(List<String> sentence) {
			int sentenceLength = sentence.size();
			
			for (int sentencePosition = 0; sentencePosition < sentenceLength; sentencePosition++) {
				String word = sentence.get(sentencePosition);
				HuffmanNode huffmanNode = huffmanNodes.get(word);

				for (int c = 0; c < layer1_size; c++)
					neu1[c] = 0;
				for (int c = 0; c < layer1_size; c++)
					neu1e[c] = 0;
				
				nextRandom = incrementRandom(nextRandom);
				int b = (int)((nextRandom % window) + window) % window;
				
				// in -> hidden                                                                                                                                                                                       
				int cw = 0;
				for (int a = b; a < window * 2 + 1 - b; a++) {
					if (a == window)
						continue;
					int c = sentencePosition - window + a;
					if (c < 0 || c >= sentenceLength)
						continue;
					int idx = huffmanNodes.get(sentence.get(c)).idx;
					for (int d = 0; d < layer1_size; d++) {
						neu1[d] += syn0[idx][d];
					}
					
					cw++;
				}
				
				if (cw == 0)
					continue;
				
				for (int c = 0; c < layer1_size; c++)
					neu1[c] /= cw;
				
				if (config.useHierarchicalSoftmax) {
					for (int d = 0; d < huffmanNode.code.length; d++) {
						double f = 0;
						int l2 = huffmanNode.point[d];
						// Propagate hidden -> output                                                                                                                                                                     
						for (int c = 0; c < layer1_size; c++)
							f += neu1[c] * syn1[l2][c];
						if (f <= -MAX_EXP || f >= MAX_EXP)
							continue;
						else
							f = EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
						// 'g' is the gradient multiplied by the learning rate                                                                                                                                            
						double g = (1 - huffmanNode.code[d] - f) * alpha;
						// Propagate errors output -> hidden                                                                                                                                                              
						for (int c = 0; c < layer1_size; c++)
							neu1e[c] += g * syn1[l2][c];
						// Learn weights hidden -> output                                                                                                                                                                 
						for (int c = 0; c < layer1_size; c++)
							syn1[l2][c] += g * neu1[c];
					}
				}
				
				handleNegativeSampling(huffmanNode);
				
				// hidden -> in                                                                                                                                                                                     
				for (int a = b; a < window * 2 + 1 - b; a++) {
					if (a == window)
						continue;
					int c = sentencePosition - window + a;
					if (c < 0 || c >= sentenceLength)
						continue;
					int idx = huffmanNodes.get(sentence.get(c)).idx;
					for (int d = 0; d < layer1_size; d++)
						syn0[idx][d] += neu1e[d];
				}
			}
		}
	}

	@Override Worker createWorker(int randomSeed, int iter, Iterable<List<String>> batch) {
		return new CBOWWorker(randomSeed, iter, batch);
	}
}
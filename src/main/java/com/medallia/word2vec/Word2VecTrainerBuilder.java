package com.medallia.word2vec;

import com.google.common.base.MoreObjects;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.Multiset;
import com.medallia.word2vec.util.AutoLog;
import org.apache.commons.logging.Log;
import com.medallia.word2vec.neuralnetwork.NeuralNetworkConfig;
import com.medallia.word2vec.neuralnetwork.NeuralNetworkType;

import java.util.List;
import java.util.Map;

/**
 * Builder pattern for training a new {@link Word2VecModel}
 * <p>
 * This is a port of the open source C implementation of word2vec
 * <p>
 * Note that this isn't a completely faithful rewrite, specifically:
 * <ul>
 * <li> When building the vocabulary from the training file:
 * 		<ul>
 * 			<li> The original version does a reduction step when learning the vocabulary from the file
 * 				when the vocab size hits 21 million words, removing any words that do not meet the
 * 				minimum frequency threshold.  This Java port has no such reduction step.
 * 			<li> The original version injects a &lt;/s&gt; token into the vocabulary (with a word count of 0)
 * 				as a substitute for newlines in the input file.  This Java port's vocabulary excludes the token.
 * 		</ul> 
 * <li> In partitioning the file for processing
 * 		<ul>
 *			<li> The original version assumes that sentences are delimited by newline characters and injects a sentence
 *				boundary per 1000 non-filtered tokens, i.e. valid token by the vocabulary and not removed by the randomized
 *				sampling process.  Java port mimics this behavior for now ...
 *			<li> When the original version encounters an empty line in the input file, it re-processes the first word of the
 *				last non-empty line with a sentence length of 0 and updates the random value.  Java port omits this behavior.
 * 		</ul>
 * <li> In the sampling function
 * 		<ul>
 * 			<li> The original C documentation indicates that the range should be between 0 and 1e-5, but the default value is 1e-3.
 * 				This Java port retains that confusing information.
 * 			<li> The random value generated for comparison to determine if a token should be filtered uses a float.
 * 				This Java port uses double precision for twice the fun
 * 		</ul>
 * <li> In the distance function to find the nearest matches to a target query
 * 		<ul>
 * 			<li> The original version includes an unnecessary normalization of the vector for the input query which
 * 				may lead to tiny inaccuracies.  This Java port foregoes this superfluous operation.
 * 			<li> The original version has an O(n * k) algorithm for finding top matches and is hardcoded to 40 matches.
 * 				This Java port uses Google's lovely {@link com.google.common.collect.Ordering#greatestOf(java.util.Iterator, int)}
 * 				which is O(n + k log k) and takes in arbitrary k.
 * 		</ul>
 * <li> The k-means clustering option is excluded in the Java port
 * </ul>
 * 
 * <p>
 * <p>
 * Please do not hesitate to peek at the source code.
 * <br>
 * It should be readable, concise, and correct.
 * <br>
 * I ask you to reach out if it is not.
 */
public class Word2VecTrainerBuilder {
	private static final Log LOG = AutoLog.getLog();
	
	private Integer layerSize;
	private Integer windowSize;
	private Integer numThreads;
	private NeuralNetworkType type;
	private int negativeSamples;
	private boolean useHierarchicalSoftmax;
	private Multiset<String> vocab;
	private Integer minFrequency;
	private Double initialLearningRate;
	private Double downSampleRate;
	private Integer iterations;
	private TrainingProgressListener listener;
	
	Word2VecTrainerBuilder() {
	}
	
	/** 
	 * Size of the layers in the neural network model
	 * <p>
	 * Defaults to 100
	 */
	public Word2VecTrainerBuilder setLayerSize(int layerSize) {
		Preconditions.checkArgument(layerSize > 0, "Value must be positive");
		this.layerSize = layerSize;
		return this;
	}
	
	/** 
	 * Size of the window to consider
	 * <p>
	 * Default window size is 5 tokens
	 */
	public Word2VecTrainerBuilder setWindowSize(int windowSize) {
		Preconditions.checkArgument(windowSize > 0, "Value must be positive");
		this.windowSize = windowSize;
		return this;
	}
	
	/** 
	 * Specify number of threads to use for parallelization
	 * <p>
	 * Defaults to {@link Runtime#availableProcessors()}
	 */
	public Word2VecTrainerBuilder useNumThreads(int numThreads) {
		Preconditions.checkArgument(numThreads > 0, "Value must be positive");
		this.numThreads = numThreads;
		return this;
	}
	
	/** 
	 * @see {@link NeuralNetworkType}
	 * <p>
	 * By default, word2vec uses the {@link NeuralNetworkType#SKIP_GRAM}
	 */
	public Word2VecTrainerBuilder type(NeuralNetworkType type) {
		this.type = Preconditions.checkNotNull(type);
		return this;
	}
	
	/** 
	 * Specify to use hierarchical softmax
	 * <p>
	 * By default, word2vec does not use hierarchical softmax
	 */
	public Word2VecTrainerBuilder useHierarchicalSoftmax() {
		this.useHierarchicalSoftmax = true;
		return this;
	}
	
	/** 
	 * Number of negative samples to use
	 * Common values are between 5 and 10
	 * <p>
	 * Defaults to 0
	 */
	public Word2VecTrainerBuilder useNegativeSamples(int negativeSamples) {
		Preconditions.checkArgument(negativeSamples >= 0, "Value must be non-negative");
		this.negativeSamples = negativeSamples;
		return this;
	}
	
	/** 
	 * Use a pre-built vocabulary
	 * <p>
	 * If this is not specified, word2vec will attempt to learn a vocabulary from the training data
	 * @param vocab {@link Map} from token to frequency
	 */
	public Word2VecTrainerBuilder useVocab(Multiset<String> vocab) {
		this.vocab = Preconditions.checkNotNull(vocab);
		return this;
	}
	
	/** 
	 * Specify the minimum frequency for a valid token to be considered
	 * part of the vocabulary
	 * <p>
	 * Defaults to 5
	 */
	public Word2VecTrainerBuilder setMinVocabFrequency(int minFrequency) {
		Preconditions.checkArgument(minFrequency >= 0, "Value must be non-negative");
		this.minFrequency = minFrequency;
		return this;
	}
	
	/**
	 * Set the starting learning rate
	 * <p>
	 * Default is 0.025 for skip-gram and 0.05 for CBOW
	 */
	public Word2VecTrainerBuilder setInitialLearningRate(double initialLearningRate) {
		Preconditions.checkArgument(initialLearningRate >= 0, "Value must be non-negative");
		this.initialLearningRate = initialLearningRate;
		return this;
	}
	
	/**
	 * Set threshold for occurrence of words. Those that appear with higher frequency in the training data,
	 * e.g. stopwords, will be randomly removed
	 * <p>
	 * Default is 1e-3, useful range is (0, 1e-5)
	 */
	public Word2VecTrainerBuilder setDownSamplingRate(double downSampleRate) {
		Preconditions.checkArgument(downSampleRate >= 0, "Value must be non-negative");
		this.downSampleRate = downSampleRate;
		return this;
	}
	
	/** Set the number of iterations */
	public Word2VecTrainerBuilder setNumIterations(int iterations) {
		Preconditions.checkArgument(iterations > 0, "Value must be positive");
		this.iterations = iterations;
		return this;
	}
	
	/** Set a progress listener */
	public Word2VecTrainerBuilder setListener(TrainingProgressListener listener) {
		this.listener = listener;
		return this;
	}
	
	/** Train the model */
	public Word2VecModel train(Iterable<List<String>> sentences) throws InterruptedException {
		this.type = MoreObjects.firstNonNull(type, NeuralNetworkType.CBOW);
		this.initialLearningRate = MoreObjects.firstNonNull(initialLearningRate, type.getDefaultInitialLearningRate());
		if (this.numThreads == null)
			this.numThreads = Runtime.getRuntime().availableProcessors();
		this.iterations = MoreObjects.firstNonNull(iterations, 5);
		this.layerSize = MoreObjects.firstNonNull(layerSize, 100);
		this.windowSize = MoreObjects.firstNonNull(windowSize, 5);
		this.downSampleRate = MoreObjects.firstNonNull(downSampleRate, 0.001);
		this.minFrequency = MoreObjects.firstNonNull(minFrequency, 5);
		this.listener = MoreObjects.firstNonNull(listener, new TrainingProgressListener() {
			@Override
			public void update(Stage stage, double progress) {
				System.out.println(String.format("Stage %s, progress %s%%", stage, progress));
			}
		});
		
		Optional<Multiset<String>> vocab = this.vocab == null
				? Optional.<Multiset<String>>absent()
				: Optional.of(this.vocab);
		
		return new Word2VecTrainer(
				minFrequency,
				vocab,
				new NeuralNetworkConfig(
						type,
						numThreads,
						iterations,
						layerSize,
						windowSize,
						negativeSamples,
						downSampleRate,
						initialLearningRate,
						useHierarchicalSoftmax
					)
			).train(LOG, listener, sentences);
	}
	
	/** Listener for model training progress */
	public interface TrainingProgressListener {
		/** Sequential stages of processing */
		enum Stage {
			ACQUIRE_VOCAB,
			FILTER_SORT_VOCAB,
			CREATE_HUFFMAN_ENCODING,
			TRAIN_NEURAL_NETWORK,
		}
		
		/** 
		 * Called during word2vec training
		 * <p>
		 * Note that this is called in a separate thread from the processing thread
		 * @param stage Current {@link Stage} of processing
		 * @param progress Progress of the current stage as a double value between 0 and 1
		 */
		void update(Stage stage, double progress);
	}
}
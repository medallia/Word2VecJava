package com.medallia.word2vec.util;

/** Extension of {@link AutoCloseable} where {@link #close()} does not throw any exception */
public interface AC extends AutoCloseable {
	@Override void close();

	/** {@link AC} that does nothing */
	AC NOTHING = new AC() {
		@Override public void close() { }
	};
}
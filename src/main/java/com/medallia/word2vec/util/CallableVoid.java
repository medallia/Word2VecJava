package com.medallia.word2vec.util;

import java.util.concurrent.Callable;

/** Utility base implementation of Callable with a Void return type. */
public abstract class CallableVoid implements Callable<Void> {

	@Override public final Void call() throws Exception {
		run();
		return null;
	}

	/** Do the actual work here instead of using {@link #call()} */
	protected abstract void run() throws Exception;

}

package com.medallia.word2vec.util;

import com.google.common.base.Preconditions;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Logger;
import org.apache.log4j.varia.NullAppender;

/**
 * Creates loggers based on the caller's class.
 */
public final class AutoLog {
	/** Prevents initialization. */
	private AutoLog() {
	}

	/** @return {@link org.apache.commons.logging.Log} based on the caller's class */
	public static Log getLog() {
		return getLog(2);
	}

	/** Make sure there is at least one appender to avoid a warning printed on stderr */
	private static class InitializeOnDemand {
		private static final boolean INIT = init();
		private static boolean init() {
			if (!Logger.getRootLogger().getAllAppenders().hasMoreElements())
				Logger.getRootLogger().addAppender(new NullAppender());
			return true;
		}
	}

	/** @return {@link org.apache.commons.logging.Log} based on the stacktrace distance to
	 * the original caller. 1= the caller to this method. 2 = the caller to the caller... etc*/
	public static Log getLog(int distance) {
		Preconditions.checkState(InitializeOnDemand.INIT);
		String callerClassName = Common.myCaller(distance).getClassName();
		try {
			return LogFactory.getLog(Class.forName(callerClassName));
		} catch (ClassNotFoundException t) {
			String err = "Class.forName on " + callerClassName + " failed";
			System.err.println(err);
			throw new IllegalStateException(err, t);
		}
	}
}

package com.medallia.word2vec.util;

import org.apache.thrift.TBase;
import org.apache.thrift.TDeserializer;
import org.apache.thrift.TException;
import org.apache.thrift.TSerializer;
import org.apache.thrift.protocol.TJSONProtocol;

/** Contains useful methods for using Thrift */
public final class ThriftUtils {
	private static final String THRIFT_CHARSET = "utf-8";

	/** Serialize a JSON-encoded thrift object */
	public static <T extends TBase> String serializeJson(T obj) throws TException {
		// Tried having a static final serializer, but it doesn't seem to be thread safe
		return new TSerializer(new TJSONProtocol.Factory()).toString(obj, THRIFT_CHARSET);
	}

	/** Deserialize a JSON-encoded thrift object */
	public static <T extends TBase> T deserializeJson(T dest, String thriftJson) throws TException {
		// Tried having a static final deserializer, but it doesn't seem to be thread safe
		new TDeserializer(new TJSONProtocol.Factory()).deserialize(dest, thriftJson, THRIFT_CHARSET);
		return dest;
	}
}

package webdata.field_encoder_decoder;


public interface Encoder {

	void reset(String dir);

	void saveResult();

	void add(int id,String data);

}

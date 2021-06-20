package service;

public class ClientMain {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Product p = new Product();
		p.setId((long) 1);
		p.setLabel("Orange");
		p.setPrice(10);
		System.out.print(p.getPrice());
	}

}

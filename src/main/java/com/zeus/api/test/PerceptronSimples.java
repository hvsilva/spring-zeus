package com.zeus.api.test;

public class PerceptronSimples {
	private double[] pesos;
	private double bias;
	private double taxaAprendizado = 0.1;

	public PerceptronSimples(int entradas) {
		pesos = new double[entradas];
		bias = 0;
	}

	public int prever(double[] entrada) {
		double soma = bias;
		for (int i = 0; i < pesos.length; i++) {
			soma += pesos[i] * entrada[i];
		}
		return soma >= 0 ? 1 : 0; // Função de ativação degrau
	}

	public void treinar(double[][] entradas, int[] saidasEsperadas, int epocas) {
		for (int epoca = 0; epoca < epocas; epoca++) {
			for (int i = 0; i < entradas.length; i++) {
				int saidaCalculada = prever(entradas[i]);
				int erro = saidasEsperadas[i] - saidaCalculada;
				for (int j = 0; j < pesos.length; j++) {
					pesos[j] += taxaAprendizado * erro * entradas[i][j];
				}
				bias += taxaAprendizado * erro;
			}
		}
	}

	public double[] getPesos() {
		return pesos;
	}

	public double getBias() {
		return bias;
	}
}
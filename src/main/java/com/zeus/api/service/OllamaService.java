package com.zeus.api.service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.HashMap;
import java.util.Map;

import org.springframework.stereotype.Service;

@Service
public class OllamaService {

    public Map<String, Object> perguntar(String prompt) {
        Map<String, Object> resultado = new HashMap<>();
        try {
            String json = String.format("""
                {
                    "model": "llama3",
                    "prompt": "%s",
                    "stream": false
                }
            """, prompt.replace("\"", "\\\""));

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("http://localhost:11434/api/generate"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(json))
                    .build();

            HttpClient client = HttpClient.newHttpClient();
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

            JSONObject jsonResponse = new JSONObject(response.body());
            resultado.put("resposta", jsonResponse.getString("response"));
            resultado.put("modelo", jsonResponse.optString("model", "llama3"));
            resultado.put("tempo", jsonResponse.optDouble("total_duration", 0));

        } catch (Exception e) {
            resultado.put("erro", "Erro ao chamar LLM local: " + e.getMessage());
        }
        return resultado;
    }
}
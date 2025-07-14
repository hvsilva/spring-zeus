package com.zeus.api.controller;

import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/transformer")
public class OllamaTransformerController {

    private final WebClient webClient = WebClient.create("http://localhost:11434");

    @PostMapping("/gerar")
    public Mono<Map> gerarTexto(@RequestBody Map<String, String> payload) {
        String prompt = payload.get("prompt");

        Map<String, Object> body = new HashMap<>();
        body.put("model", "llama2");
        body.put("prompt", prompt);
        body.put("stream", false);

        return webClient.post()
                .uri("/api/generate")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(Map.class);
    }
}
package com.zeus.api.controller;

import java.util.Map;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.zeus.api.service.OllamaService;

@RestController
@RequestMapping("/ollama")
public class OllamaController {

    private final OllamaService ollamaService;

    public OllamaController(OllamaService ollamaService) {
        this.ollamaService = ollamaService;
    }

    @GetMapping("/responder")
    public Map<String, Object> responder(@RequestParam String pergunta) {
        return ollamaService.perguntar(pergunta);
    }
}
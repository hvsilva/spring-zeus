package com.zeus.api.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.zeus.api.service.OpenAiServiceWrapper;

@RestController
@RequestMapping("/openai")
public class OpenAiController {
//
//    private final OpenAiServiceWrapper service;
//
//    public OpenAiController(OpenAiServiceWrapper service) {
//        this.service = service;
//    }
//
//    @GetMapping("/responder")
//    public String responder(@RequestParam String pergunta) {
//        return service.perguntar(pergunta);
//    }
}
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/joho/godotenv"
)

type EmbeddingRequest struct {
	Input interface{} `json:"input"`
	Model string      `json:"model,omitempty"`
}

type EmbeddingResult struct {
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
	Object    string    `json:"object"`
}

type EmbeddingResponse struct {
	Data   []EmbeddingResult `json:"data"`
	Model  string            `json:"model"`
	Object string            `json:"object"`
	Usage  struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

func getEnv(key, fallback string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return fallback
}

func embeddingHandlerFactory(maxBatchSize int, ovhBatchApiUrl string, ovhToken string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {

		fmt.Printf("Received request for embeddings length: %d\n", r.ContentLength)

		// Only handle POST requests
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			fmt.Printf("Method not allowed: %s\n", r.Method)
			return
		}

		var req EmbeddingRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		if err != nil {
			http.Error(w, fmt.Sprintf("Error parsing request: %v", err), http.StatusBadRequest)
			fmt.Printf("Error parsing request: %v\n", err)
			return
		}

		// Convert input to slice of texts
		var texts []string
		switch v := req.Input.(type) {
		case string:
			texts = []string{v}
		case []interface{}:
			texts = make([]string, len(v))
			for i, item := range v {
				if str, ok := item.(string); ok {
					texts[i] = str
				} else {
					texts[i] = fmt.Sprintf("%v", item)
				}
			}
		default:
			texts = []string{fmt.Sprintf("%v", v)}
		}

		// Process in batches
		var allEmbeddings [][]float64
		for i := 0; i < len(texts); i += maxBatchSize {
			end := i + maxBatchSize
			if end > len(texts) {
				end = len(texts)
			}
			batch := texts[i:end]

			fmt.Printf("Processing batch %d/%d, size: %d\n", i/maxBatchSize+1, (len(texts)+maxBatchSize-1)/maxBatchSize, len(batch))

			// Send batch request to OVH
			batchJSON, err := json.Marshal(batch)
			if err != nil {
				http.Error(w, fmt.Sprintf("Error marshaling batch: %v", err), http.StatusInternalServerError)
				fmt.Printf("Error marshaling batch: %v\n", err)
				return
			}

			req, err := http.NewRequest("POST", ovhBatchApiUrl, bytes.NewBuffer(batchJSON))
			if err != nil {
				http.Error(w, fmt.Sprintf("Error creating request: %v", err), http.StatusInternalServerError)
				fmt.Printf("Error creating request: %v\n", err)
				return
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", ovhToken))

			client := &http.Client{}
			resp, err := client.Do(req)
			if err != nil {
				http.Error(w, fmt.Sprintf("Error calling OVH API: %v", err), http.StatusInternalServerError)
				fmt.Printf("Error calling OVH API: %v\n", err)
				return
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				bodyBytes, _ := ioutil.ReadAll(resp.Body)
				http.Error(w, fmt.Sprintf("Error from OVH API (batch starting at index %d): %d, response: %s", i, resp.StatusCode, string(bodyBytes)), http.StatusInternalServerError)
				fmt.Printf("Error from OVH API (batch starting at index %d): %d, response: %s\n", i, resp.StatusCode, string(bodyBytes))
				return
			}

			var batchEmbeddings [][]float64
			err = json.NewDecoder(resp.Body).Decode(&batchEmbeddings)
			if err != nil {
				http.Error(w, fmt.Sprintf("Error decoding OVH response: %v", err), http.StatusInternalServerError)
				fmt.Printf("Error decoding OVH response: %v\n", err)
				return
			}

			allEmbeddings = append(allEmbeddings, batchEmbeddings...)
		}

		// Format response to match OpenAI's format
		embeddingsResults := make([]EmbeddingResult, len(allEmbeddings))
		for i, embedding := range allEmbeddings {
			embeddingsResults[i] = EmbeddingResult{
				Embedding: embedding,
				Index:     i,
				Object:    "embedding",
			}
		}

		// Count tokens (simple approximation by word count)
		totalTokens := 0
		for _, text := range texts {
			totalTokens += len(strings.Fields(text))
		}

		response := EmbeddingResponse{
			Data:   embeddingsResults,
			Model:  "ovh-embeddings",
			Object: "list",
		}
		response.Usage.PromptTokens = totalTokens
		response.Usage.TotalTokens = totalTokens

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		fmt.Println("Successfully processed request")
	}
}
func main() {
	// Load .env file
	err := godotenv.Load()
	if err != nil {
		log.Println("Warning: Error loading .env file:", err)
	} else {
		log.Println("Successfully loaded .env file")
	}

	maxBatchSize, err := strconv.Atoi(getEnv("BATCH_SIZE", "10"))
	if err != nil {
		panic("Error parsing OVH_BATCH_MAX_BATCH_SIZE:" + err.Error())
	}
	ovhBatchApiUrl := os.Getenv("OVH_BATCH_API_URL")

	if ovhBatchApiUrl == "" {
		panic("OVH_BATCH_API_URL not set")
	}

	// Get OVH token from environment variable
	ovhToken := os.Getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN")
	if ovhToken == "" {
		panic("OVH token not set\n")
	}

	http.HandleFunc("/v1/embeddings", embeddingHandlerFactory(maxBatchSize, ovhBatchApiUrl, ovhToken))
	port := getEnv("PORT", "14152")
	fmt.Println("Server starting on port ", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

import express from "express";
import dotenv from "dotenv";
import { ChatGroq } from "@langchain/groq";
import fs from 'fs'
import { PDFParse } from "pdf-parse";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
dotenv.config();

//google embedding
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";

import { QdrantVectorStore } from "@langchain/qdrant";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const app = express();
const PORT = process.env.PORT || 5000;

app.use(express.json());

const llm = new ChatGroq({
  model: "llama-3.3-70b-versatile",

});

//emdedding
const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001", // 768 dimensions
  taskType: TaskType.RETRIEVAL_DOCUMENT,
  title: "Document title",
});

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
  url: process.env.QDRANT_URL,
  collectionName: "store-testing",
});


const upload=async()=>
{
  const filepath="./kknn.pdf"
  const buffer=fs.readFileSync(filepath)
  const pdfresult=new PDFParse({data:buffer})
  const result=await pdfresult.getText()
  const text=result.text
  const spilitter=new RecursiveCharacterTextSplitter({
    chunkSize:500,
    chunkOverlap:250
  })
  const docs=await spilitter.createDocuments([text])
  await vectorStore.addDocuments(docs)
  
}
upload();



app.post("/ai", async (req, res) => {
  const { input } = req.body;
  const docs=await vectorStore.similaritySearch(input,5)
  const context=docs.map((d)=>d.pageContent).join("/n")
  const response=await llm.invoke([

    new SystemMessage(`You are a RAG AI assistant 
     STRICT RULES:
      -Answer ONLY from context.
      -Do not use outside knowledge.
      -If the answer is not found, say: 
      "I don't know from uploaded PDF."
      
      CONTEXT:
      ${context}`),
      new HumanMessage(input)
  ])
  console.log(docs);
  
  return res.status(200).json({"ai:":response.content});
});
app.get("/", (req, res) => {
  res.send("Hello");
});

app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});

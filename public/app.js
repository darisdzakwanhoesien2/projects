// Load Express module
const express = require('express');

// Create an Express application instance
const app = express();

// Define the port your server will listen on
const PORT = 5002;

// Set up a basic GET route that sends a welcome message
app.get('/', (req, res) => {
  res.send('Hello World!');
});

// Start the server and listen on the defined port
app.listen(PORT, (error) => {
  if (error) {
    console.log("Error occurred, server can't start", error);
  } else {
    console.log(`Server is Successfully Running, and App is listening on port ${PORT}`);
  }
});
const functions = require("firebase-functions");
const express = require("express");
const fs = require("fs");
const path = require("path");

const app = express();

// Set the view engine to EJS
app.set("view engine", "ejs");

// Serve static files from 'public'
app.use(express.static("public"));

// Define your routes
app.get("/", (req, res) => {
  const packagesFolder = path.join(process.cwd(), "public", "packages");

  fs.readdir(packagesFolder, (err, packageFolders) => {
    if (err) throw err;

    const packages = packageFolders.filter((folder) =>
      fs.statSync(path.join(packagesFolder, folder)).isDirectory()
    );

    res.render("index", {
      packages: packages,
    });
  });
});

// Serve the selected transcript JSON file dynamically
app.get("/transcripts/:packageName", (req, res) => {
  const packageName = req.params.packageName;
  const transcriptPath = path.join(
    process.cwd(),
    "public",
    "packages",
    packageName,
    "transcription_result.json"
  );

  fs.readFile(transcriptPath, "utf8", (err, data) => {
    if (err) {
      return res.status(404).send({ error: "Transcript not found." });
    }
    res.json(JSON.parse(data));
  });
});

// Export the app as a Firebase Function
exports.app = functions.https.onRequest(app);

// const functions = require("firebase-functions");
// const express = require("express");
// const fs = require("fs");
// const path = require("path");

// const app = express();

// // Set the view engine to EJS
// app.set("view engine", "ejs");

// // Serve static files from 'public'
// app.use(express.static("public"));

// // Define your routes
// app.get("/", (req, res) => {
//   const packagesFolder = path.join(__dirname, "public", "packages");

//   fs.readdir(packagesFolder, (err, packageFolders) => {
//     if (err) throw err;

//     const packages = packageFolders.filter((folder) =>
//       fs.statSync(path.join(packagesFolder, folder)).isDirectory()
//     );

//     res.render("index", {
//       packages: packages,
//     });
//   });
// });

// // Serve the selected transcript JSON file dynamically
// app.get("/transcripts/:packageName", (req, res) => {
//   const packageName = req.params.packageName;
//   const transcriptPath = path.join(
//     __dirname,
//     "public",
//     "packages",
//     packageName,
//     "transcription_result.json"
//   );

//   fs.readFile(transcriptPath, "utf8", (err, data) => {
//     if (err) {
//       return res.status(404).send({error: "Transcript not found."});
//     }
//     res.json(JSON.parse(data));
//   });
// });

// // Export the app as a Firebase Function
// exports.app = functions.https.onRequest(app);


// const functions = require("firebase-functions");
// const express = require("express");
// const fs = require("fs");
// const path = require("path");

// const app = express();

// // Set the view engine to EJS
// app.set("view engine", "ejs");

// // Serve static files from 'public'
// app.use(express.static("public"));

// // Define your routes
// app.get("/", (req, res) => {
//   const packagesFolder = path.join(__dirname, "public", "packages");

//   fs.readdir(packagesFolder, (err, packageFolders) => {
//     if (err) throw err;

//     const packages = packageFolders.filter((folder) =>
//       fs.statSync(path.join(packagesFolder, folder)).isDirectory()
//     );

//     res.render("index", {
//       packages: packages,
//     });
//   });
// });

// // Serve the selected transcript JSON file dynamically
// app.get("/transcripts/:packageName", (req, res) => {
//   const packageName = req.params.packageName;
//   const transcriptPath = path.join(
//     __dirname,
//     "public",
//     "packages",
//     packageName,
//     "transcription_result.json"
//   );

//   fs.readFile(transcriptPath, "utf8", (err, data) => {
//     if (err) {
//       return res.status(404).send({error: "Transcript not found."});
//     }
//     res.json(JSON.parse(data));
//   });
// });

// // Export the app as a Firebase Function
// exports.app = functions.https.onRequest(app);


// const functions = require("firebase-functions");
// const express = require("express");
// const fs = require("fs");
// const path = require("path");

// const app = express();

// // Set the view engine to EJS
// app.set("view engine", "ejs");

// // Serve static files from 'public'
// app.use(express.static("public"));

// // Define your routes
// app.get("/", (req, res) => {
//   const packagesFolder = path.join(__dirname, "public", "packages");

//   fs.readdir(packagesFolder, (err, packageFolders) => {
//     if (err) throw err;

//     const packages = packageFolders.filter((folder) =>
//       fs.statSync(path.join(packagesFolder, folder)).isDirectory()
//     );

//     res.render("index", {
//       packages: packages,
//     });    
//   });
// });

// // Serve the selected transcript JSON file dynamically
// app.get("/transcripts/:packageName", (req, res) => {
//   const packageName = req.params.packageName;
//   const transcriptPath = path.join(
//     __dirname,
//     "public",
//     "packages",
//     packageName,
//     "transcription_result.json"
//   );
  

//   fs.readFile(transcriptPath, "utf8", (err, data) => {
//     if (err) {
//       return res.status(404).send({error: "Transcript not found."});
//     }
//     res.json(JSON.parse(data));
//   });
// });

// // Export the app as a Firebase Function
// exports.app = functions.https.onRequest(app);


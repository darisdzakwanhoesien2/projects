const express = require('express');
const fs = require('fs');
const app = express();
const port = 3000;

const path = require('path');

// Set the view engine to EJS
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));


// Get all package folders from 'public/packages'
app.get('/', (req, res) => {
  const packagesFolder = path.join(__dirname, 'public', 'packages');

  // Read all the folders inside the 'packages' directory
  fs.readdir(packagesFolder, (err, packageFolders) => {
    if (err) throw err;

    // Filter only directories (each directory represents a package)
    const packages = packageFolders.filter(folder => 
      fs.statSync(path.join(packagesFolder, folder)).isDirectory());

    res.render('index', {
      packages: packages
    });
  });
});

// Serve static files from 'public'
app.use(express.static('public'));

// Serve the selected transcript JSON file dynamically
app.get('/transcripts/:packageName', (req, res) => {
  const packageName = req.params.packageName;
  const transcriptPath = path.join(__dirname, 'public', 'packages', packageName, 'transcription_result.json');

  // Send the transcript JSON file
  fs.readFile(transcriptPath, 'utf8', (err, data) => {
    if (err) {
      return res.status(404).send({ error: 'Transcript not found.' });
    }
    res.json(JSON.parse(data));
  });
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});

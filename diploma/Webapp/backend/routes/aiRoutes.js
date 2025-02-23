const { spawn } = require("child_process");
const express = require("express");
const router = express.Router();

router.post("/predict", (req, res) => {
    const { features } = req.body;

    if (!features || !Array.isArray(features)) {
        return res.status(400).json({ error: "Invalid input format. 'features' must be an array." });
    }

    const pythonProcess = spawn("python", ["cardio.py"]);

    pythonProcess.stdin.write(JSON.stringify({ features }));  
    pythonProcess.stdin.end();

    let output = "";
    let errorOutput = "";

    pythonProcess.stdout.on("data", (data) => {
        output += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
        errorOutput += data.toString();
        console.error("Error from Python script:", data.toString());
    });

    pythonProcess.on("close", (code) => {
        console.log(`Python process exited with code ${code}`);

        if (errorOutput) {
            return res.status(500).json({ error: "Error from Python script", details: errorOutput });
        }

        try {
            const result = JSON.parse(output);
            res.json(result);
        } catch (error) {
            console.error("Error parsing Python output:", output);
            res.status(500).json({ error: "Error processing prediction.", details: output });
        }
    });
});

module.exports = router;

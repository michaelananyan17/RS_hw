avascript
// Global variables
let model;
let isTraining = false;
let currentEpoch = 0;
let totalEpochs = 5;

// Initialize application when window loads
window.onload = async function() {
    try {
        // Update status
        updateStatus('Loading movie data...');
        
        // Load data first
        const data = await loadData();
        
        // Update global variables with loaded data
        movies = data.movies;
        ratings = data.ratings;
        numUsers = data.numUsers;
        numMovies = data.numMovies;
        
        // Populate dropdowns
        populateUserDropdown();
        populateMovieDropdown();
        
        // Update status and start training
        updateStatus('Data loaded. Training model...');
        
        // Train the model
        await trainModel();
        
    } catch (error) {
        console.error('Initialization error:', error);
        updateStatus('Error initializing application: ' + error.message, true);
    }
};

function populateUserDropdown() {
    const userSelect = document.getElementById('user-select');
    userSelect.innerHTML = '';
    
    // Get unique user IDs from the ratings data
    const userIds = [...new Set(ratings.map(r => r.userId))].sort((a, b) => a - b);
    
    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Select a user';
    userSelect.appendChild(defaultOption);
    
    // Add users to the dropdown
    userIds.forEach(userId => {
        const option = document.createElement('option');
        option.value = userId;
        option.textContent = `User ${userId}`;
        userSelect.appendChild(option);
    });
}

function populateMovieDropdown() {
    const movieSelect = document.getElementById('movie-select');
    movieSelect.innerHTML = '';
    
    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Select a movie';
    movieSelect.appendChild(defaultOption);
    
    // Add movies sorted by title
    movies.sort((a, b) => a.title.localeCompare(b.title)).forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.year ? `${movie.title} (${movie.year})` : movie.title;
        movieSelect.appendChild(option);
    });
}

async function trainModel() {
    try {
        isTraining = true;
        currentEpoch = 0;
        
        // Update status to show we're starting training
        updateStatus('Starting model training...');
        updateProgressBar(0);
        updateEpochInfo('Epoch: 0/' + totalEpochs);
        
        // Define model architecture for matrix factorization
        const numLatentFactors = 10; // Increased for better performance
        
        // User embedding layer
        const userEmbedding = tf.layers.embedding({
            inputDim: numUsers + 1, // +1 for 1-based indexing
            outputDim: numLatentFactors,
            embeddingsInitializer: 'randomNormal',
            name: 'userEmbedding'
        });
        
        // Movie embedding layer
        const movieEmbedding = tf.layers.embedding({
            inputDim: numMovies + 1, // +1 for 1-based indexing
            outputDim: numLatentFactors,
            embeddingsInitializer: 'randomNormal',
            name: 'movieEmbedding'
        });

        // Input layers
        const userInput = tf.layers.input({ shape: [1], name: 'userInput' });
        const movieInput = tf.layers.input({ shape: [1], name: 'movieInput' });

        // Apply embeddings
        const userVec = userEmbedding.apply(userInput);
        const movieVec = movieEmbedding.apply(movieInput);
        
        // Flatten the embeddings
        const userFlat = tf.layers.flatten().apply(userVec);
        const movieFlat = tf.layers.flatten().apply(movieVec);

        // Dot product of user and movie vectors
        const dotProduct = tf.layers.dot({ axes: 1 }).apply([userFlat, movieFlat]);
        
        // Add bias terms
        const userBias = tf.layers.embedding({
            inputDim: numUsers + 1,
            outputDim: 1,
            name: 'userBias'
        }).apply(userInput);
        
        const movieBias = tf.layers.embedding({
            inputDim: numMovies + 1,
            outputDim: 1,
            name: 'movieBias'
        }).apply(movieInput);
        
        const userBiasFlat = tf.layers.flatten().apply(userBias);
        const movieBiasFlat = tf.layers.flatten().apply(movieBias);
        
        // Combine dot product with biases
        const addLayer = tf.layers.add().apply([dotProduct, userBiasFlat, movieBiasFlat]);
        
        // Create model
        model = tf.model({ 
            inputs: [userInput, movieInput], 
            outputs: addLayer 
        });
        
        // Compile model with lower learning rate for better convergence
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });

        // Prepare training data
        const userTensor = tf.tensor1d(ratings.map(r => r.userId), 'int32');
        const movieTensor = tf.tensor1d(ratings.map(r => r.movieId), 'int32');
        const ratingTensor = tf.tensor1d(ratings.map(r => r.rating), 'float32');

        // Train the model
        await model.fit([userTensor, movieTensor], ratingTensor, {
            epochs: totalEpochs,
            batchSize: 32,
            validationSplit: 0.1,
            shuffle: true,
            callbacks: {
                onEpochBegin: (epoch) => {
                    currentEpoch = epoch;
                    const progress = ((epoch) / totalEpochs) * 100;
                    updateProgressBar(progress);
                    updateEpochInfo(`Epoch: ${epoch + 1}/${totalEpochs}`);
                },
                onEpochEnd: (epoch, logs) => {
                    const progress = ((epoch + 1) / totalEpochs) * 100;
                    updateProgressBar(progress);
                    updateStatus(`Epoch ${epoch + 1}/${totalEpochs} - Loss: ${logs.loss.toFixed(4)}${logs.val_loss ? `, Val Loss: ${logs.val_loss.toFixed(4)}` : ''}`);
                    console.log(`Epoch ${epoch + 1}, Loss: ${logs.loss}`);
                }
            }
        });
        
        // Clean up tensors
        tf.dispose([userTensor, movieTensor, ratingTensor]);

        updateStatus('Model training complete! Ready for predictions.');
        updateProgressBar(100);
        updateEpochInfo('Training complete!');
        document.getElementById('predict-btn').disabled = false;
        
        // Log model summary
        console.log('Model training completed successfully');
        model.summary();
        
    } catch (error) {
        console.error('Training error:', error);
        updateStatus('Error training model: ' + error.message, true);
    } finally {
        isTraining = false;
    }
}

async function predictRating() {
    if (!model || isTraining) {
        updateResult('Model is not ready yet. Please wait for training to complete.', 'low');
        return;
    }

    try {
        const userId = parseInt(document.getElementById('user-select').value);
        const movieId = parseInt(document.getElementById('movie-select').value);

        if (isNaN(userId) || isNaN(movieId) || userId === 0 || movieId === 0) {
            updateResult('Please select both a user and a movie.', 'low');
            return;
        }

        // Show loading state
        updateResult('<div class="loading"></div> Making prediction...', 'medium');

        // Create input tensors
        const userTensor = tf.tensor1d([userId], 'int32');
        const movieTensor = tf.tensor1d([movieId], 'int32');
        
        // Make prediction
        const prediction = model.predict([userTensor, movieTensor]);
        const ratingArray = await prediction.data();
        let predictedRating = ratingArray[0];
        
        // Ensure rating is within 1-5 range
        predictedRating = Math.max(1, Math.min(5, predictedRating));
        
        // Clean up tensors
        tf.dispose([userTensor, movieTensor, prediction]);
        
        // Display result
        const movie = movies.find(m => m.id === movieId);
        const movieTitle = movie ? (movie.year ? `${movie.title} (${movie.year})` : movie.title) : `Movie ${movieId}`;
        
        // Check if this user has already rated this movie
        const existingRating = ratings.find(r => r.userId === userId && r.movieId === movieId);
        
        let ratingClass = 'medium';
        if (predictedRating >= 4) ratingClass = 'high';
        else if (predictedRating <= 2) ratingClass = 'low';
        
        let resultHTML = `Predicted rating for <strong>User ${userId}</strong> on "<strong>${movieTitle}</strong>": <strong>${predictedRating.toFixed(2)}/5</strong>`;
        
        if (existingRating) {
            resultHTML += `<br><small>Actual rating: ${existingRating.rating}/5</small>`;
        }
        
        updateResult(resultHTML, ratingClass);
        
    } catch (error) {
        console.error('Prediction error:', error);
        updateResult('Error making prediction: ' + error.message, 'low');
    }
}

// UI helper functions
function updateStatus(message, isError = false) {
    const statusElement = document.getElementById('status');
    statusElement.textContent = message;
    statusElement.style.borderLeftColor = isError ? '#e74c3c' : '#3498db';
    statusElement.style.background = isError ? '#fdedec' : '#f8f9fa';
}

function updateResult(message, className = '') {
    const resultElement = document.getElementById('result');
    resultElement.innerHTML = message;
    resultElement.className = `result ${className}`;
}

function updateProgressBar(percentage) {
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.width = `${percentage}%`;
}

function updateEpochInfo(message) {
    const epochInfo = document.getElementById('epoch-info');
    epochInfo.textContent = message;
}

// Add some utility functions for better user experience
function getUserStats(userId) {
    const userRatings = ratings.filter(r => r.userId === userId);
    const avgRating = userRatings.reduce((sum, r) => sum + r.rating, 0) / userRatings.length;
    return {
        totalRatings: userRatings.length,
        averageRating: avgRating.toFixed(2)
    };
}

function getMovieStats(movieId) {
    const movieRatings = ratings.filter(r => r.movieId === movieId);
    const avgRating = movieRatings.reduce((sum, r) => sum + r.rating, 0) / movieRatings.length;
    return {
        totalRatings: movieRatings.length,
        averageRating: avgRating.toFixed(2)
    };
}

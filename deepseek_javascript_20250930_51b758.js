async function trainModel() {
    try {
        isTraining = true;
        
        // Define model architecture
        const numLatentFactors = 8;
        
        // Calculate max IDs for embedding layers
        const maxUserId = Math.max(...ratings.map(r => r.userId));
        const maxMovieId = Math.max(...movies.map(m => m.id));
        
        const userEmbedding = tf.layers.embedding({
            inputDim: maxUserId + 1,  // Fixed: use max ID + 1
            outputDim: numLatentFactors,
            inputLength: 1,
            name: 'user-embedding'
        });
        
        const movieEmbedding = tf.layers.embedding({
            inputDim: maxMovieId + 1,  // Fixed: use max ID + 1
            outputDim: numLatentFactors,
            inputLength: 1,
            name: 'movie-embedding'
        });

        const userInput = tf.layers.input({ shape: [1], name: 'user-input' });
        const movieInput = tf.layers.input({ shape: [1], name: 'movie-input' });

        const userVec = userEmbedding.apply(userInput);
        const movieVec = movieEmbedding.apply(movieInput);
        
        const userFlatten = tf.layers.flatten().apply(userVec);
        const movieFlatten = tf.layers.flatten().apply(movieVec);

        const dotProduct = tf.layers.dot({ axes: 1 }).apply([userFlatten, movieFlatten]);
        
        model = tf.model({ inputs: [userInput, movieInput], outputs: dotProduct });
        model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

        // Prepare data for training
        const userTensors = tf.tensor2d(ratings.map(r => r.userId), [ratings.length, 1]);
        const movieTensors = tf.tensor2d(ratings.map(r => r.movieId), [ratings.length, 1]);
        const ratingTensors = tf.tensor2d(ratings.map(r => r.rating), [ratings.length, 1]);

        // Train the model
        await model.fit([userTensors, movieTensors], ratingTensors, {
            epochs: 5,
            batchSize: 64,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    updateStatus(`Epoch ${epoch + 1}/5, Loss: ${logs.loss.toFixed(4)}`);
                }
            }
        });
        
        // Clean up tensors
        tf.dispose([userTensors, movieTensors, ratingTensors]);

        updateStatus('Model training complete!');
        document.getElementById('predict-btn').disabled = false;
        
    } catch (error) {
        console.error('Training error:', error);
        updateStatus('Error training model: ' + error.message, true);
    } finally {
        isTraining = false;
    }
}
function AudioManager(){
    var currentAudio = null;
    this.wavesurfer = null;
    this.initSurfer = function(){
        this.wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: '#D1DA58',
            progressColor: '#E25141',
            normalize:true
        });
    }
    this.onUpload = function(file){
        if(file){
            currentAudio = file;
            this.drawAudio(file);
        }
    }
    this.playAudio = function(){
        this.wavesurfer.playPause()
    }
    this.drawAudio = function(file){
        this.wavesurfer.loadBlob(file);
    }
    this.hidePredictions = function (){
        document.getElementById('predictions').classList.add('hidden')
    }
    function updatePredictions(classes, probs){
        document.getElementById('predictions').classList.remove('hidden')
        for(var i = 0; i < classes.length; i++){
            classEl = document.getElementById('class-'+(i+1)).innerText = classes[i]
            probEl = document.getElementById('prob-'+(i+1)).innerText = (probs[i]*100).toFixed(2) + "%"
        }
    }
    this.tagAudio = function(){
        this.hidePredictions()
        document.getElementById('spinner').classList.add('is-active')
        var fd = new FormData();
        fd.append('audio',currentAudio);
        fetch('/label_file', { 
          method: 'POST',
          body:fd
        })
        .then(resp => resp.json(),resp => console.log(resp))
        .catch(function (error){
            console.log("couldn't parse json!",error);
            document.getElementById('spinner').classList.remove('is-active')
        })
        .then(function(response){
            classes=response['classes'][0]
            probs = response['probabilities'][0]
            updatePredictions(classes, probs)
            document.getElementById('spinner').classList.remove('is-active')
        })
    }
}
audioManager = new AudioManager()
document.addEventListener("DOMContentLoaded", function(){
    audioManager.initSurfer();
    audioManager.hidePredictions()
});
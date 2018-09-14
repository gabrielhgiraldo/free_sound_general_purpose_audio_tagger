function AudioManager(){
    var currentAudio = null;
    this.wavesurfer = null;
    this.initSurfer = function(){
        this.wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: 'violet',
            progressColor: 'purple',
            normalize:true
        });
    }
    this.onUpload = function(file){
        currentAudio = file;
        this.drawAudio(file);
    }
    this.playAudio = function(){
        this.wavesurfer.playPause()
    }
    this.drawAudio = function(file){
        this.wavesurfer.loadBlob(file);
    }
    this.tagAudio = function(){
        var fd = new FormData();
        fd.append('audio',currentAudio);
        fetch('/label_file', { 
          method: 'POST',
          body:fd
        })
        .then(resp => resp.json(),resp => console.log(resp))
        .catch(function (error){
            console.log("couldn't parse json!",error);
        })
        .then(function(response){
            document.getElementById('predictions').innerText=response['classes']
        })
    }
}
audioManager = new AudioManager()
document.addEventListener("DOMContentLoaded", function(){
    audioManager.initSurfer();
});
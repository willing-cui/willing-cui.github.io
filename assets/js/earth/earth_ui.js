import {earth} from "./earth.js";

const mapMarkerButton = document.getElementById("mapMarkerButton");
mapMarkerButton.addEventListener("click", () => { enableMapMarker(); });
var mapMarkerOn = 1;

var mapMarkerButtonStyle = document.querySelector('#mapMarkerButton').style;
mapMarkerButtonStyle.setProperty('--background-color', 'rgba(255, 255, 255, 0.15)');

function enableMapMarker() {
    if (mapMarkerOn) {
        mapMarkerButtonStyle.setProperty('--background-color', 'rgba(255, 255, 255, 0.0)');
        earth.earthGroup.remove(earth.landmarkGroup);
        mapMarkerOn = 0;
    } else {
        mapMarkerButtonStyle.setProperty('--background-color', 'rgba(255, 255, 255, 0.15)');
        earth.earthGroup.add(earth.landmarkGroup);
        mapMarkerOn = 1;
    }
}
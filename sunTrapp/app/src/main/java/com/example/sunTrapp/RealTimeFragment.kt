package com.example.sunTrapp

//import com.google.android.gms.location.LocationServices
//import com.google.android.gms.location.FusedLocationProviderClient
//import com.google.android.gms.location.LocationCallback
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.drawable.BitmapDrawable
import android.graphics.drawable.Drawable
import android.location.Location
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.content.res.AppCompatResources
import androidx.core.content.res.ResourcesCompat
import androidx.core.graphics.drawable.toBitmap
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.google.android.gms.location.*
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.GoogleMap
import com.google.android.gms.maps.MapView
import com.google.android.gms.maps.OnMapReadyCallback
import com.google.android.gms.maps.model.BitmapDescriptorFactory
import com.google.android.gms.maps.model.LatLng
import com.google.android.gms.maps.model.MarkerOptions
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import android.widget.Button
import com.google.android.gms.maps.CameraUpdate
import java.util.*
import java.io.File
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

import com.bumptech.glide.Glide
import android.os.AsyncTask
import android.widget.EditText
import android.view.inputmethod.InputMethodManager
import androidx.core.content.ContextCompat.getSystemService

class RealTimeFragment : Fragment() {
    /**
     * TextView showing Longitude. We are keeping this as variable in order to prevent to ask the UI for this textView
     * everytime we want to update the UI
     */
    private lateinit var tvLong : TextView

    /**
     * TextView showing Latitude. We are keeping this as variable in order to prevent to ask the UI for this textView
     * everytime we want to update the UI
     */
    private lateinit var tvLat : TextView

    /**
     * TextView showing Altitude. We are keeping this as variable in order to prevent to ask the UI for this textView
     * everytime we want to update the UI
     */
    private lateinit var tvAlt : TextView

    private lateinit var tvDis : TextView
    private lateinit var tvDis2 : TextView
    private lateinit var tvDis3 : TextView
    private lateinit var textSearch : EditText
    private  lateinit var buttonSearch : Button


    private lateinit var textHour : EditText
    private lateinit var textMinute : EditText
    private lateinit var textDay : EditText
    private lateinit var textMonth : EditText


    //    private lateinit var pyplotImage : WebView
    private lateinit var pyplotImage : ImageView
//    private lateinit var pyplotImage : pl.droidsonroids.gif.GifImageView


    private lateinit var tvTreeCard1_dist : MutableList<TextView>
    private lateinit var tvTreeCard1_bear : MutableList<TextView>
    private lateinit var tvTreeCard1_spec : MutableList<TextView>
    private lateinit var tvTreeCard1_local : MutableList<TextView>
    private lateinit var tvTreeCard1_public : MutableList<TextView>
    private lateinit var tvTreeCard1_link : MutableList<TextView>
    private lateinit var tvTreeCard1_image : MutableList<ImageView>
    private lateinit var tvTreeCard1_compass : MutableList<ImageView>

    var latest_position = LatLng(51.4545, -2.5879)
//    private lateinit var tvTreeCard1_cardView : MutableList<CardView>
    private lateinit var tvTreeCard1_viewButton : MutableList<Button>

    private var current_closest_gpsLocations = mutableListOf<GPSLocation>()

    private var counter = 0
    private var counter_refresh = 0
    private var selectedOption : String = "All species"
    private var selectedOption_award : String = "All trees"

    private lateinit var mapView: MapView
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var locationCallback: LocationCallback

    private lateinit var current_position: LatLng
    private lateinit var current_CameraUpdate: CameraUpdate
    /**
     * It tells if we should center the camera of the map every time a position is retrieved. If the user has moved the map, we want to keep
     * the settings made to the map by the user themselves instead of re-centering
     */
    private var moveCamera = true
    private var pause_updates = false

    /**
     * Reference to the map shown on the screen
     */
    private lateinit var map : GoogleMap

    /**
     * ActivityViewModel shared with Activity and Fragment
     */
    private val model: ActivityViewModel by activityViewModels()

    /**
     * True if the map has already been initialized, false otherwise
     */
    private var initialized = false

    /**
     * Variable used to display just one toast at once.
     * When we the user clicks on the "Center" button multiple times, this would cause many toasts to queue up. Therefore,
     * we are saving the current toast and if the user generates another toast, we cancel the current toast and display a new one.
     * This way, there will always be one toast in the queue
     */
    private var toast: Toast? = null

    private val receiverData : BroadcastReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            Log.d("ReceiverSample","ReceivedData")

            //If the activity is bound to the user, we update the ui
            if(model.mBound) {
                updateUI()
            }
        }
    }

    /**
     * Updates the Interface: So both the value displayed on the cards and on the map
     */
    private fun updateUI()
    {
        if(!model.mBound)
            return

//        val sample = model.readerService!!.currentSample
        //Update the 3 cards shown at the top of the screen
//        updateCards(sample)
//        //If latitude and longitude are valid
//        if (sample.latitude != null && sample.longitude != null) {
//            //if the map has been initialized
//            if(initialized) {
//                //Insert the marker with the current position of the user
//                insertMarker()
//                //Center the camera to where the marker is placed if we should do so (the user has not moved the map camera)
//                if (moveCamera) {
//                    moveCameraToCurrentPosition()
//                }
//            }
//        }
//        else if(initialized){
//            //if a valid position is not available, we will not show anything on the map. The information about NoData available
//            //is handled already by the function updateCards()
//            map.clear()
//        }
    }

    /**
     * Inserts a marker in the current position if the app is collecting locations.
     *
     */
    private fun insertMarker(){
        if(model.mBound && model.readerService!!.isCollectingLocation && initialized) {

            val sample = model.readerService!!.currentSample
            //Latitude and longitude cannot be null if the app is collecting location. There is no need to check
            val pos = LatLng(sample.latitude!!.toDouble(), sample.longitude!!.toDouble())

            //Creating the icon object to show as marker
            var bitmap =
                AppCompatResources.getDrawable(requireContext(), R.drawable.ic_marker3)!!.toBitmap()
            bitmap = Bitmap.createScaledBitmap(bitmap, 130, 130, false)
            val icon = BitmapDescriptorFactory.fromBitmap(bitmap)

            //Creating the marker
            val marker = MarkerOptions()
                .position(pos)
                .title(getString(R.string.you_are_here))
                .snippet("Lat:" + pos.latitude.toString() + ", Lng:" + pos.longitude.toString())
                .icon(icon)
            //Clear the marker previously positioned
            map.clear()
            //Add the new marker to the map
            map.addMarker(marker)
        }
    }

    private val callback = OnMapReadyCallback { googleMap ->

        /**
         * Manipulates the map once available.
         * This callback is triggered when the map is ready to be used.
         * This is where we can add markers or lines, add listeners or move the camera.
         * In this case, we just add a marker near Sydney, Australia.
         * If Google Play services is not installed on the device, the user will be prompted to
         * install it inside the SupportMapFragment. This method will only be triggered once the
         * user has installed Google Play services and returned to the app.
         */

        map = googleMap

        //Disabling zoom +/- controls at the bottom right-hand side of the screen
        map.uiSettings.isZoomControlsEnabled = false
        //The user cannot tilt the map
        map.uiSettings.isTiltGesturesEnabled = false

        map.setOnCameraMoveStartedListener {
            //If the camera has been moved by the user with a gesture, we have to stop to recenter the map every time a new location
            //is available
            if (it == GoogleMap.OnCameraMoveStartedListener.REASON_GESTURE) {
                Log.d("moveCamera", "Camera moving")
                moveCamera = false
            }
        }

        googleMap.setOnMapClickListener { latLng ->
            // Create a marker at the clicked location
            val markerOptions = MarkerOptions().position(latLng).title("Clicked Location")
            googleMap.clear()
            googleMap.addMarker(markerOptions)
            val loc = LocationDetails(latLng.longitude.toString(), latLng.latitude.toString(), "0.", Date(System.currentTimeMillis()))
            pause_updates = false

            val pos = LatLng(latLng.latitude.toDouble(), latLng.longitude.toDouble())
            val update = CameraUpdateFactory.newLatLngZoom(pos, 16f)
            map.animateCamera(update)

            updateCards(latLng.longitude.toString(), latLng.latitude.toString())
        }


        val pos = LatLng(51.4545, -2.5879)
        val update = CameraUpdateFactory.newLatLngZoom(pos, 12f)
        map.moveCamera(update)

        //insert the marker in the current position
        insertMarker()

        //The map has been initialized
        initialized = true

        updateUI()
    }

    /**
     * When the fragments is in the foreground and receives the input from the user we subscribe for updates from the service again and
     * we update the UI if possible.
     *
     */
    override fun onResume() {
        super.onResume()

        //Using LocalBroadcastManager instead of simple BroadcastManager. This has several advantages:
        // - You know that the data you are broadcasting won't leave your app, so don't need to worry about leaking private data.
        // - It is not possible for other applications to send these broadcasts to your app, so you don't need to worry about having security holes they can exploit.
        // - It is more efficient than sending a global broadcast through the system.
        //we are subscribing for updates about the latest sample
//        LocalBroadcastManager.getInstance(requireContext()).registerReceiver(receiverData, IntentFilter(ReaderService.ACTION_NEW_SAMPLE))

        //We want to update the UI but move the camera to current position without any animation (animate set to false)
//        updateUI()
        mapView.onResume()
    }
override fun onDestroy() {
        super.onDestroy()
        // Important: Call the MapView's onDestroy() method
        mapView.onDestroy()
    }

    override fun onLowMemory() {
        super.onLowMemory()
        // Important: Call the MapView's onLowMemory() method
        mapView.onLowMemory()
    }
    /**
     * Moves the camera to the last retrieved position in the readerService
     */
    private fun moveCameraToCurrentPosition(){
        val sample = model.readerService!!.currentSample
        if(sample.latitude != null && sample.longitude != null) {
            val pos = LatLng(sample.latitude.toDouble(), sample.longitude.toDouble())
            val update = CameraUpdateFactory.newLatLngZoom(pos, 15f)
            //We cannot use animateCamera() as this would cause many messages I/Counters: exceeded sample count in FrameTime
            //in the debug log
            map.moveCamera(update)
        }
    }

    /**
     * UnRegisters the receiver when the fragment is not visible, there is no need to keep receiving the updates if we do not need to update the
     * interface.
     */
    override fun onPause() {
        super.onPause()

        LocalBroadcastManager.getInstance(requireContext()).unregisterReceiver(receiverData)
        //remove pending broadcasts, they must not be handled when the receiver is restarted
        receiverData.abortBroadcast
        mapView.onPause()
    }

    private fun clearGoogleMapsMarkers(){
        mapView.getMapAsync { googleMap ->
            googleMap.clear() // Clear previous markers if any
        }
    }

    private fun updateMapWithLocation(latitude: Double, longitude: Double, red_label: Boolean = false) {
        val currentLocation = LatLng(latitude, longitude)

        val height = 60 // resize according to your zooming level
        val width = 60 // resize according to your zooming level
        var bitmapDraw: BitmapDrawable
        var loc_string: String
        if(red_label){
            bitmapDraw = resources.getDrawable(R.drawable.red_dot) as BitmapDrawable
            loc_string = "Pinned Location"
        }
        else{
            bitmapDraw = resources.getDrawable(R.drawable.blue_dot) as BitmapDrawable
            loc_string = "My Location"
        }
        val bitmap = bitmapDraw.bitmap
        val finalMarker = Bitmap.createScaledBitmap(bitmap, width, height, false)

        mapView.getMapAsync { googleMap ->
//            googleMap.clear() // Clear previous markers if any
            googleMap.addMarker(MarkerOptions().position(currentLocation).icon(BitmapDescriptorFactory.fromBitmap(finalMarker)).title(loc_string).zIndex(9f))
//            googleMap.animateCamera(CameraUpdateFactory.newLatLngZoom(currentLocation, 15f))
        }
    }

    private fun updateMapWithTreeLocation(latitude: Double, longitude: Double, species: String = "", alpha: Float = 1f) {
        val currentLocation = LatLng(latitude, longitude)

        val height = 150 // resize according to your zooming level
        val width = 150 // resize according to your zooming level
        val bitmapDraw = resources.getDrawable(R.drawable.tree) as BitmapDrawable
        val bitmap = bitmapDraw.bitmap
        val finalMarker = Bitmap.createScaledBitmap(bitmap, width, height, false)

        mapView.getMapAsync { googleMap ->
            val marker = googleMap.addMarker(MarkerOptions().position(currentLocation).icon(BitmapDescriptorFactory.fromBitmap(finalMarker)).title(species).alpha(alpha))
        }



    }



    /**
     * Creates the view the first time the fragment is created.
     *
     * @param inflater
     * @param container
     * @param savedInstanceState
     * @return
     */
    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?,
                              savedInstanceState : Bundle?): View?{
        //Inflating layout from the resources
        val view = inflater.inflate(R.layout.fragment_real_time, container,false)

        fun View.hideKeyboard() {
            val inputManager = requireContext().getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
            inputManager.hideSoftInputFromWindow(windowToken, 0)
        }


        mapView = view.findViewById(R.id.map_view)
        mapView.onCreate(savedInstanceState)
        mapView.getMapAsync(callback)



        val button: Button = view.findViewById(R.id.map_centre_button)
        button.setOnClickListener {
//            map.animateCamera(CameraUpdateFactory.newLatLngZoom(current_position, 15f))
            try{
//                map.animateCamera(current_CameraUpdate)
                val pos = LatLng(51.4545, -2.5879)
                val update = CameraUpdateFactory.newLatLngZoom(pos, 12f)
                map.moveCamera(update)
            }
            catch(e: Exception){}

        }

        pyplotImage = view.findViewById(R.id.image_view_pyplot)

        textSearch = view.findViewById<EditText>(R.id.editTextSearch)

        textHour = view.findViewById<EditText>(R.id.editTextHour)
        textMinute = view.findViewById<EditText>(R.id.editTextMinute)
        textDay = view.findViewById<EditText>(R.id.editTextDay)
        textMonth = view.findViewById<EditText>(R.id.editTextMonth)


        val buttonSearch = view.findViewById<Button>(R.id.buttonSearch)

        buttonSearch.setOnClickListener {
            val searchText = textSearch.text.toString()


            var hourText = textHour.text.toString()
            var minuteText = textMinute.text.toString()
            var dayText = textDay.text.toString()
            var monthText = textMonth.text.toString()

            val py = Python.getInstance()
            val module = py.getModule("plot")

            val result = module.callAttr("query_google_maps_search", searchText)
            val resultList = result.asList()


            var pos: LatLng
            if(searchText.isEmpty()){
                updateCards(latest_position.latitude.toString(), latest_position.longitude.toString(), searchText, hourText, minuteText, dayText, monthText)
                pos = LatLng(latest_position.longitude, latest_position.latitude)
            }
            else{
                updateCards(resultList[1].toString(), resultList[0].toString(), searchText, hourText, minuteText, dayText, monthText)
                pos = LatLng(resultList[0].toDouble(), resultList[1].toDouble())
            }

            val update = CameraUpdateFactory.newLatLngZoom(pos, 16f)
            map.animateCamera(update)

            val markerOptions = MarkerOptions().position(pos).title("Searched Location")
            map.clear()
            map.addMarker(markerOptions)


            // hide the keyboard
//            val inputMethodManager = requireContext().getSystemService(requireContext().INPUT_METHOD_SERVICE) as InputMethodManager
//            inputMethodManager.hideSoftInputFromWindow(textSearch.windowToken, 0)
            it.hideKeyboard()
        }




//        val html = """
//            <html>
//            <body>
//            <img src="file:///android_asset/output_Seneca.gif" alt="Animated GIF">
//            </body>
//            </html>
//        """.trimIndent()
//        pyplotImage.loadDataWithBaseURL(null, html, "text/html", "UTF-8", null)


        if (! Python.isStarted()) {
            Python.start(AndroidPlatform(requireContext()))
        }
//        val py = Python.getInstance()
//        val module = py.getModule("plot")
//        val bytes = module.callAttr("plot_func").toJava(ByteArray::class.java)
//        // // plot image
////        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
////        pyplotImage.setImageBitmap(bitmap)
//
//        // // Load the animated GIF using Glide
//        val filePath = File(requireContext().filesDir, "gif_file.gif").absolutePath
//        Glide.with(requireContext()).asGif().load(filePath).into(pyplotImage)
//





        return view
    }



    data class GPSLocation(val latitude: Double, val longitude: Double, val species: String,
                           val localName: String, val veteranStatus: String, val publicAccessibilityStatus: String,
                            val TNSI: String, val heritageTree: String, val TotY: String, val championTree: String,
                           val treeID: String
                           )




    private fun calculateDistance(location1: GPSLocation, location2: GPSLocation): Double {
        val earthRadius = 6371 // Radius of the earth in kilometers
        val latDifference = Math.toRadians(location2.latitude - location1.latitude)
        val lonDifference = Math.toRadians(location2.longitude - location1.longitude)
        val a = sin(latDifference / 2) * sin(latDifference / 2) +
                cos(Math.toRadians(location1.latitude)) * cos(Math.toRadians(location2.latitude)) *
                sin(lonDifference / 2) * sin(lonDifference / 2)
        val c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return earthRadius * c
    }

    private fun findClosestLocations(referenceLocation: GPSLocation, allLocations: List<GPSLocation>, count: Int): List<GPSLocation> {
        val sortedLocations = allLocations.sortedBy { calculateDistance(referenceLocation, it) }
        return sortedLocations.subList(0, minOf(count, sortedLocations.size))
    }

    private fun readGPSLocationsFromAssets(context: Context, fileName: String, referenceLocation: GPSLocation): List<GPSLocation> {
        val gpsLocations = mutableListOf<GPSLocation>()

//        counter = 0
        val listOfDistances = MutableList(7) { index -> 1000000.0 }

        for (i in 0..1) {
//            val fileName_i = "trees_full_part_"+i.toString().dropLast(2)+".txt"
            val fileName_i = "trees_full_part_"+i.toString()+".txt"
//            val fileName_i = "trees_full_part_1.txt"
            val inputStream = context.assets.open(fileName_i)

        try {


            val size = inputStream.available()
            val buffer = ByteArray(size)
            inputStream.read(buffer)
            inputStream.close()

            val fileContent = String(buffer, Charsets.UTF_8)
            val lines = fileContent.split("\n")

            for (line in lines) {

                val parts = line.split(",")

                val species = parts[2].toString()
                val TNSI = parts[6].toString()
                val heritageTree = parts[7].toString()
                val TotY = parts[8].toString()
                val championTree = parts[9].toString()

                var special_tree = "False"
                if (TNSI != "False" || heritageTree != "False" || TotY != "False" || championTree != "False") {
                    special_tree = "True"
                }

                if ((selectedOption == "All species") || (species == selectedOption)) {
                    if ((selectedOption_award == "All trees") || ((selectedOption_award == "Only special trees") && (special_tree == "True"))) {

                        val latitude = parts[0].toDouble()
                        val longitude = parts[1].toDouble()
                        val localName = parts[3].toString()
                        val veteranStatus = parts[4].toString()
                        val publicAccessibilityStatus = parts[5].toString()

                        val treeID = parts[10].toString()
                        val gpsLocation = GPSLocation(
                            latitude, longitude, species,
                            localName, veteranStatus, publicAccessibilityStatus, TNSI,
                            heritageTree, TotY, championTree, treeID
                        )

                        val distance = calculateDistance(referenceLocation, gpsLocation)
                        val maxDistanceIndex = listOfDistances.indexOf(listOfDistances.maxOrNull())
                        if (distance < listOfDistances[maxDistanceIndex]) {
                            if (gpsLocations.size < 7) {
                                gpsLocations.add(gpsLocation)
                            } else {
                                gpsLocations[maxDistanceIndex] = gpsLocation
                            }
                            listOfDistances[maxDistanceIndex] = distance
                        }

//                        gpsLocations.add(gpsLocation)
//                        counter++
                    }
                }


            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

        return gpsLocations
    }

    private fun calculateBearing(location1: Location, location2: Location): Float {
        return location1.bearingTo(location2)
    }

    private fun roundBearingToNearestTenDegrees(bearing: Double): Double {
        val degrees = bearing % 360.0
        val remainder = degrees % 10.0

        return when {
            remainder < 5.0 -> degrees - remainder
            else -> degrees + (10.0 - remainder)
        }
    }


    /**
     * Update the cards showing the last positions after an update has been sent by the service.
     *
     * @param location location details to update the cards with. If any of the 3 components is null, that means that it was impossible to retrieve
     * the location. Therefore, we show that No data is available.
     */
    private fun updateCards(latitude_in: String, longitude_in: String, display_string: String="", hourText: String="", minuteText: String="", dayText: String="", monthText: String="")
    {
        counter = counter + 1
        latest_position = LatLng(latitude_in.toDouble(), longitude_in.toDouble())

        // UPDATES GO HERE

        // // Load the animated GIF using Glide
//        val loading_gif = resources.getDrawable(R.drawable.loading)
//        val filePath = File(requireContext().filesDir, "gif_file.gif").absolutePath

        val loading_gif = resources.getIdentifier("loading", "drawable", requireContext().packageName)
        Glide.with(requireContext()).asGif().load(loading_gif).into(pyplotImage)

        // Execute Python code in a background thread using AsyncTask
        object : AsyncTask<Void, Void, ByteArray>() {
            override fun doInBackground(vararg params: Void?): ByteArray {
                val py = Python.getInstance()
                val module = py.getModule("plot")
                return module.callAttr("plot_func", counter, latitude_in, longitude_in, display_string, hourText, minuteText, dayText, monthText).toJava(ByteArray::class.java)
            }

            override fun onPostExecute(result: ByteArray?) {
                super.onPostExecute(result)
                // Update the ImageView with the result image on the UI thread
                if (result != null) {
                    val filePath = File(requireContext().filesDir, "gif_file_"+latitude_in+"_"+longitude_in+"_"+counter.toString()+".gif").absolutePath
                    Glide.with(requireContext())
                        .asGif()
                        .load(filePath)
                        .into(pyplotImage)
                }
            }
        }.execute()

//        val py = Python.getInstance()
//        val module = py.getModule("plot")
//        val bytes = module.callAttr("plot_func").toJava(ByteArray::class.java)
//        // // plot image
////        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
////        pyplotImage.setImageBitmap(bitmap)
//
//        // // Load the animated GIF using Glide
//        val filePath = File(requireContext().filesDir, "gif_file.gif").absolutePath
//        Glide.with(requireContext()).asGif().load(filePath).into(pyplotImage)

    }



}


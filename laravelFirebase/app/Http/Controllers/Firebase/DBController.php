<?php

namespace App\Http\Controllers\Firebase;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Kreait\Firebase\Contract\Database;
use GuzzleHttp\Client;

class DBController extends Controller
{
    public function __construct(Database $database)
    {
        $this->database = $database;
        $this->tablename = 'data';
    }

    public function index()
    {
        $client = new Client([
            'verify' => false, 
        ]);

        try {
            $response = $client->get('https://esp32-b8e97-default-rtdb.asia-southeast1.firebasedatabase.app/' . $this->tablename . '.json');
            $infom = json_decode($response->getBody()->getContents(), true);
            return view('layout.app', compact('infom'));
        } catch (\GuzzleHttp\Exception\RequestException $e) {
            
        }
    }
}

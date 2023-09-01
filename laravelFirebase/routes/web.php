<?php

use App\Http\Controllers\Firebase\DBController;
use Illuminate\Support\Facades\Route;

Route::get('/', [DBController::class, 'index']);

Route::get('/db', function () {
    return view('welcome');
});


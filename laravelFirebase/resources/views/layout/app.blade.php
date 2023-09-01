<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="{{ mix('css/app.css') }}">
    @vite(['resources/css/app.css'])
</head>


<body class="flex items-center flex-col justify-between min-h-screen bg-gray-100">

  {{-- Header --}}
  <div class="flex flex-row w-full h-fit justify-center items-center py-4 bg-[#3F4D77] text-white">
    <div class="container flex w-full justify-between items-center flex-row px-4">
      <div class="flex flex-row gap-2">
        <div class="h-11 w-full">
            <img class="w-full h-full object-contain" src="{{ asset('assets/images/polije.png') }}"
              alt="polije">
        </div>
        <div class="h-11 w-full">
          <img class="w-full h-full object-contain" src="{{ asset('assets/images/knu.png') }}" alt="polije">
        </div>
      </div>
      <div class="flex flex-col text-sm md:text-lg">
        <p class="font-poppins-semibold">Smart and Safe Street Lamp</p>
      </div>
    </div>
  </div>
  {{-- Header End--}}


  {{-- First Section --}}
    <div class="flex w-full h-full flex-1 container flex-col px-4">
        <div class="flex flex-col justify-normal mt-8 h-fit w-full gap-6 bg-white py-4 px-6 border-2 shadow-sm">
            
            <h1 class=" text-xl md:text-3xl h-fit mt-4 text-center font-poppins-semibold">Title Here</h1>
            <div
                class="w-full overflow-y-auto gap-8 h-fit grid-flow-row min-h-fit rounded-xl grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 pb-6 pt-2">
                

                {{-- test card --}}
                @forelse (array_reverse($infom) as $key => $item)
                    <div class="border-2 shadow-sm rounded-md h-fit flex flex-col w-full grid-flow-row">

                        <div class="h-64 w-full rounded-md overflow-hidden">
                            <img class="w-full h-full object-cover" src="{{ $item['image'] }}" alt="pic">
                        </div>
                        
                        <p class="py-2 px-2 text-base md:text-xl text-center"><span class="font-bold">
                          {{$item['timestamp']}}
                        <p class="px-2 py-2 line-clamp-3 text-base text-justify">
                          {{$item['info']}}
                        </p>

                    </div>
                    @empty
                    @endforelse
                {{-- test card --}}

            </div>
            {{-- <div class="flex h-fit flex-1 flex-col bg-white rounded-xl border-2 shadow-sm px-6 gap-3 py-8">
                <h1 class="font-bold text-xl">Information <span class="text-red-700">*</span></h1>
                <p>1. Photo</p>
                <p>2. Time</p>
                <p>3. Information</p>
            </div> --}}
        </div>
    </div>
    {{-- End of First Section --}}


    {{-- Second Section --}}
    <div class="flex flex-col justify-center items-center w-full py-2 mt-6 bg-gray-800 text-white  text-center px-30 md:px-0">
      <div class="flex w-full h-full flex-1 container flex-col px-4">
        <div class="flex flex-col justify-normal mt-8 h-fit w-full gap-6 py-2 px-20 md:px-36" >
          <h2 class="font-poppins-semibold my-6 text-xl md:text-3xl">Meet the Team </h2>
      
          {{-- First Group --}}
          <div class="w-full overflow-y-auto gap-10 h-fit grid-flow-row min-h-fit rounded-xl grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 pb-6 pt-2">
          {{-- First Person --}}
            <div class="border-1 shadow-sm rounded-md h-fit flex flex-col w-full grid-flow-row justify-center items-center">
              <div class="h-36 w-36 rounded-full overflow-hidden ">
                <img class="w-full h-full object-cover" src="{{ asset('assets/images/eunji.jpg') }}" alt="Eu-Ji Kim">
              </div>
              <p class="py-2 px-2 text-base md:text-xl text-center font-poppins-bold">Eun-Ji Kim
              <p class="px-2 py-0 text-xs md:text-sm text-center">Kyungpook National University</p>
              <div class="flex flex-row items-center justify-center py-4 px-2 gap-5">
                <a href="https://github.com/EunJiKim02" >
                  <!-- GitHub -->
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
                    class="h-6 w-6 text-primary dark:text-primary-400">
                    <path fill="currentColor"
                      d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                  </svg>
                </a>
                <a href="https://www.linkedin.com/" >
                  <!-- Linkedin -->
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
                    class="h-4 w-4 text-primary dark:text-primary-400">
                    <path fill="currentColor"
                              d="M4.98 3.5c0 1.381-1.11 2.5-2.48 2.5s-2.48-1.119-2.48-2.5c0-1.38 1.11-2.5 2.48-2.5s2.48 1.12 2.48 2.5zm.02 4.5h-5v16h5v-16zm7.982 0h-4.968v16h4.969v-8.399c0-4.67 6.029-5.052 6.029 0v8.399h4.988v-10.131c0-7.88-8.922-7.593-11.018-3.714v-2.155z" />              </svg>
                </a>
                </div>
            </div>
           {{-- End of First Person --}}

           {{-- Second Person --}}
           <div class="border-1 shadow-sm rounded-md h-fit flex flex-col w-full grid-flow-row justify-center items-center">
            <div class="h-36 w-36 rounded-full overflow-hidden ">
                <img class="w-full h-full object-cover" src="{{ asset('assets/images/dongje.jpg') }}" alt="pic">
              </div>
              <p class="py-2 px-2 text-base md:text-xl text-center font-poppins-bold">Dong-Je Park
              <p class="px-2 py-0 text-xs md:text-sm  text-center">Kyungpook National University</p>
              <div class="flex flex-row items-center justify-center py-4 px-2 gap-5">
                <a href="https://github.com/" >
                  <!-- GitHub -->
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
                    class="h-6 w-6 text-primary dark:text-primary-400">
                    <path fill="currentColor"
                      d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                  </svg>
                </a>
                <a href="https://www.linkedin.com" >
                  <!-- Linkedin -->
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
                    class="h-4 w-4 text-primary dark:text-primary-400">
                    <path fill="currentColor"
                              d="M4.98 3.5c0 1.381-1.11 2.5-2.48 2.5s-2.48-1.119-2.48-2.5c0-1.38 1.11-2.5 2.48-2.5s2.48 1.12 2.48 2.5zm.02 4.5h-5v16h5v-16zm7.982 0h-4.968v16h4.969v-8.399c0-4.67 6.029-5.052 6.029 0v8.399h4.988v-10.131c0-7.88-8.922-7.593-11.018-3.714v-2.155z" />              </svg>
                </a>
                </div>
            </div>
           {{-- End of Second Person --}}

           {{-- Third Person--}}
           <div class="border-1 shadow-sm rounded-md h-fit flex flex-col w-full grid-flow-row justify-center items-center">
            <div class="h-36 w-36 rounded-full overflow-hidden ">
                <img class="w-full h-full object-cover" src="{{ asset('assets/images/seonghee.png') }}" alt="pic">
              </div>
              <p class="py-2 px-2 text-base md:text-xl text-center font-poppins-bold">Seong-Hee Gu
              <p class="px-2 py-0 text-xs md:text-sm  text-center">Kyungpook National University</p>
              <div class="flex flex-row items-center justify-center py-4 px-2 gap-5">
                <a href="https://github.com/" >
                  <!-- GitHub -->
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
                    class="h-6 w-6 text-primary dark:text-primary-400">
                    <path fill="currentColor"
                      d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                  </svg>
                </a>
                <a href="https://www.linkedin.com" >
                  <!-- Linkedin -->
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
                    class="h-4 w-4 text-primary dark:text-primary-400">
                    <path fill="currentColor"
                              d="M4.98 3.5c0 1.381-1.11 2.5-2.48 2.5s-2.48-1.119-2.48-2.5c0-1.38 1.11-2.5 2.48-2.5s2.48 1.12 2.48 2.5zm.02 4.5h-5v16h5v-16zm7.982 0h-4.968v16h4.969v-8.399c0-4.67 6.029-5.052 6.029 0v8.399h4.988v-10.131c0-7.88-8.922-7.593-11.018-3.714v-2.155z" />              </svg>
                </a>
                </div>
            </div>
           {{-- End of Third Person --}}
          </div>
          {{-- End of First Group --}}

          {{-- Second Group --}}
          <div class="w-full overflow-y-auto gap-10 h-fit grid-flow-row min-h-fit rounded-xl grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 pb-6 pt-2">
           {{-- Fourth Person --}}
           <div class="border-1 shadow-sm rounded-md h-fit flex flex-col w-full grid-flow-row justify-center items-center">
            <div class="h-36 w-36 rounded-full overflow-hidden ">
              <img class="w-full h-full object-cover" src="{{ asset('assets/images/jo.jpeg') }}" alt="pic">
            </div>
            <p class="py-2 px-2 text-base md:text-xl text-center font-poppins-bold">Johardio Eka 
            <p class="px-2 py-0 text-xs md:text-sm  text-center">Politeknik Negeri Jember</p>
            <div class="flex flex-row items-center justify-center py-4 px-2 gap-5">
              <a href="https://github.com/johardioeka" >
                <!-- GitHub -->
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
                  class="h-6 w-6 text-primary dark:text-primary-400">
                  <path fill="currentColor"
                    d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                </svg>
              </a>
              <a href="https://www.linkedin.com/" >
                <!-- Linkedin -->
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
                  class="h-4 w-4 text-primary dark:text-primary-400">
                  <path fill="currentColor"
                            d="M4.98 3.5c0 1.381-1.11 2.5-2.48 2.5s-2.48-1.119-2.48-2.5c0-1.38 1.11-2.5 2.48-2.5s2.48 1.12 2.48 2.5zm.02 4.5h-5v16h5v-16zm7.982 0h-4.968v16h4.969v-8.399c0-4.67 6.029-5.052 6.029 0v8.399h4.988v-10.131c0-7.88-8.922-7.593-11.018-3.714v-2.155z" />              </svg>
              </a>
              </div>
          </div>
         {{-- End of Fourth Person --}}

          {{-- Fifth Person --}}
          <div class="border-1 shadow-sm rounded-md h-fit flex flex-col w-full grid-flow-row justify-center items-center">
            <div class="h-36 w-36 rounded-full overflow-hidden ">
              <img class="w-full h-full object-cover" src="{{ asset('assets/images/aya.jpeg') }}" alt="pic">
            </div>
            <p class="py-2 px-2 text-base md:text-xl text-center font-poppins-bold">Athiyah
            <p class="px-2 py-0 text-xs md:text-sm  text-center">Politeknik Negeri Jember</p>
            <div class="flex flex-row items-center justify-center py-4 px-2 gap-5">
            <a href="https://github.com/icyaa" >
              <!-- GitHub -->
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
                class="h-6 w-6 text-primary dark:text-primary-400">
                <path fill="currentColor"
                  d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
              </svg>
            </a>
            <a href="https://www.linkedin.com/in/thisathiyah/" >
              <!-- Linkedin -->
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
                class="h-4 w-4 text-primary dark:text-primary-400">
                <path fill="currentColor"
                          d="M4.98 3.5c0 1.381-1.11 2.5-2.48 2.5s-2.48-1.119-2.48-2.5c0-1.38 1.11-2.5 2.48-2.5s2.48 1.12 2.48 2.5zm.02 4.5h-5v16h5v-16zm7.982 0h-4.968v16h4.969v-8.399c0-4.67 6.029-5.052 6.029 0v8.399h4.988v-10.131c0-7.88-8.922-7.593-11.018-3.714v-2.155z" />              </svg>
            </a>
            </div>
          </div>
         {{-- End of Fifth Person --}}

          {{-- Last Person --}}
          <div class="border-1 shadow-sm rounded-md h-fit flex flex-col w-full grid-flow-row justify-center items-center">
            <div class="h-36 w-36 rounded-full overflow-hidden ">
              <img class="w-full h-full object-cover" src="{{ asset('assets/images/dim.jpg') }}" alt="pic">
            </div>
            <p class="py-2 px-2 text-base md:text-xl text-center font-poppins-bold">Dimaz Raditya M
            <p class="px-2 py-0 text-xs md:text-sm  text-center">Politeknik Negeri Jember</p>
            <div class="flex flex-row items-center justify-center py-4 px-2 gap-5">
              <a href="https://github.com/dimazrm" >
                <!-- GitHub -->
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
                  class="h-6 w-6 text-primary dark:text-primary-400">
                  <path fill="currentColor"
                    d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                </svg>
              </a>
              <a href="https://www.linkedin.com/" >
                <!-- Linkedin -->
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
                  class="h-4 w-4 text-primary dark:text-primary-400">
                  <path fill="currentColor"
                            d="M4.98 3.5c0 1.381-1.11 2.5-2.48 2.5s-2.48-1.119-2.48-2.5c0-1.38 1.11-2.5 2.48-2.5s2.48 1.12 2.48 2.5zm.02 4.5h-5v16h5v-16zm7.982 0h-4.968v16h4.969v-8.399c0-4.67 6.029-5.052 6.029 0v8.399h4.988v-10.131c0-7.88-8.922-7.593-11.018-3.714v-2.155z" />              </svg>
              </a>
              </div>
          </div>
         {{-- End of Last Person --}}
          </div>
         
        </div>
      </div>

      <footer>
        <div class="text-center py-3 mt-10 px-4 text-xs">
            <p class="text-white">Copyright &copy 2023 ICT Volunteer Program Team C</p>
            <p class="mt-1 text-white">All Right Reserved  </p>
        </div>
    </footer>
    </div> 

    {{-- End of Second Section --}}
  <script type="module" src="{{ mix('resources/js/app.js') }}"></script>

  </body>

</html>

#include "pr.hpp"
#include <iostream>
#include <memory>
#include <map>
#include <Orochi/Orochi.h>

namespace orox
{
    enum class CompileMode
    {
        Release,
        Debug
    };
    inline void loadAsVector(std::vector<char>* buffer, const char* fllePath)
    {
        FILE* fp = fopen(fllePath, "rb");
        if (fp == nullptr)
        {
            return;
        }

        fseek(fp, 0, SEEK_END);

        buffer->resize(ftell(fp));

        fseek(fp, 0, SEEK_SET);

        size_t s = fread(buffer->data(), 1, buffer->size(), fp);
        if (s != buffer->size())
        {
            buffer->clear();
            return;
        }
        fclose(fp);
        fp = nullptr;
    }

    struct ShaderArgument
    {
        template <class T>
        void add( T p )
        {
            int bytes = sizeof( p );
            int location = m_buffer.size();
            m_buffer.resize( m_buffer.size() + bytes );
            memcpy(m_buffer.data() + location, &p, bytes );
            m_locations.push_back(location);
        }
        void clear()
        {
            m_buffer.clear();
            m_locations.clear();
        }

        std::vector<void*> kernelParams() const
        {
            std::vector<void*> ps;
            for (int i = 0; i < m_locations.size(); i++)
            {
                ps.push_back( (void * )( m_buffer.data() + m_locations[i] ) );
            }
            return ps;
        }
    private:
        std::vector<char> m_buffer;
        std::vector<int> m_locations;
    };
    class Shader
    {
    public:
        Shader( const char* filename, const std::vector<std::string>& includeDirs, const std::vector<std::string>& extraArgs, CompileMode compileMode )
        {
            std::vector<char> src;
            loadAsVector( &src, filename );
            src.push_back('\0');

            orortcProgram program = 0;
            orortcCreateProgram( &program, src.data(), "testKernel", 0, 0, 0 );
            std::vector<std::string> options;
            for( int i = 0 ; i < includeDirs.size() ; ++i)
            {
                options.push_back("-I " + includeDirs[i]);
            }

            if( compileMode == CompileMode::Debug )
            {
                options.push_back("-G");
            }

            std::vector<const char*> optionChars;
            for (int i = 0; i < options.size(); ++i)
            {
                optionChars.push_back(options[i].c_str());
            }
            
            orortcResult compileResult = orortcCompileProgram( program, optionChars.size(), optionChars.data() );

            size_t logSize = 0;
            orortcGetProgramLogSize(program, &logSize);
            if( 1 < logSize )
            {
                std::vector<char> compileLog( logSize );
                orortcGetProgramLog( program, compileLog.data() );
                printf("%s", compileLog.data());
            }
            PR_ASSERT( compileResult == ORORTC_SUCCESS );

            size_t codeSize = 0;
            orortcGetCodeSize(program, &codeSize);

            std::vector<char> codec(codeSize);
            orortcGetCode( program, codec.data() );

            orortcDestroyProgram( &program );

            orortcResult re;
            oroError e = oroModuleLoadData( &m_module, codec.data() );
            PR_ASSERT( e == oroSuccess );
        }
        ~Shader()
        {
            oroModuleUnload( m_module );
        }
        void launch( const char* name, 
            const ShaderArgument &arguments,
            unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, 
            unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, 
            oroStream hStream )
        {
            if( m_functions.count( name ) == 0 )
            {
                oroFunction f = 0;
                oroError e = oroModuleGetFunction( &f, m_module, name );
                PR_ASSERT(e == oroSuccess);
                m_functions[name] = f;
            }

            auto params = arguments.kernelParams();
            oroFunction f = m_functions[name];
            oroError e = oroModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, hStream, params.data(), 0);
            PR_ASSERT(e == oroSuccess);
        }
    private:
        oroModule m_module = 0;
        std::map<std::string, oroFunction> m_functions;
    };
}
int main() {
    using namespace pr;

    if (oroInitialize((oroApi)(ORO_API_HIP | ORO_API_CUDA), 0))
    {
        printf("failed to init..\n");
        return 0;
    }

    SetDataDir(ExecutableDir());

    oroError err;
    err = oroInit(0);
    oroDevice device;
    err = oroDeviceGet(&device, 0);
    oroCtx ctx;
    err = oroCtxCreate(&ctx, 0, device);
    oroCtxSetCurrent(ctx);

    oroStream stream = 0;
    oroStreamCreate(&stream);

    oroDeviceProp props;
    oroGetDeviceProperties(&props, device);
    printf("GPU: %s\n", props.name);

    orox::Shader shader(GetDataPath("kernels/something.cu").c_str(), {}, {}, orox::CompileMode::Release);

    int NBlocks = 128;
    oroDeviceptr bufferA = 0;
    oroDeviceptr bufferB = 0;
    oroMalloc(&bufferA, 64 * NBlocks * sizeof(float));
    oroMalloc(&bufferB, 4 * sizeof(float));

    for (;;)
    {
        oroMemsetD8Async( bufferA, 0, 64 * NBlocks * sizeof(float), stream );
        float vs[] = { 1, 2, 3, 4 };
        oroMemcpyHtoDAsync( bufferB, vs, 4 * sizeof(float), stream );

        orox::ShaderArgument args;
        args.add(bufferA);
        args.add(bufferB);
        shader.launch( "hoge", args, NBlocks, 1, 1, 64, 1, 1, stream );

        oroStreamSynchronize( stream );

        std::vector<float> result( 64 * NBlocks);
        oroMemcpyDtoH(result.data(), bufferA, 64 * NBlocks * sizeof(float));

        for (int i = 0; i < 64 * NBlocks ; i++)
        {
            PR_ASSERT( result[i] == 10.0f );
        }
    }

    
  //  Config config;
  //  config.ScreenWidth = 1920;
  //  config.ScreenHeight = 1080;
  //  config.SwapInterval = 1;
  //  Initialize(config);

  //  Camera3D camera;
  //  camera.origin = { 4, 4, 4 };
  //  camera.lookat = { 0, 0, 0 };
  //  camera.zUp = true;

  //  double e = GetElapsedTime();

  //  while (pr::NextFrame() == false) {
  //      if (IsImGuiUsingMouse() == false) {
  //          UpdateCameraBlenderLike(&camera);
  //      }

  //      ClearBackground(0.1f, 0.1f, 0.1f, 1);

  //      BeginCamera(camera);

  //      PushGraphicState();

  //      DrawGrid(GridAxis::XY, 1.0f, 10, { 128, 128, 128 });
  //      DrawXYZAxis(1.0f);

  //      static glm::vec3 P = { 0, 0, 1 };
		//ManipulatePosition(camera, &P, 0.3f);

  //      DrawSphere(P, 1.0f, { 255, 255, 255 });

  //      PopGraphicState();
  //      EndCamera();

  //      BeginImGui();

  //      ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
  //      ImGui::Begin("Panel");
  //      ImGui::Text("fps = %f", GetFrameRate());

  //      ImGui::End();

  //      EndImGui();
  //  }

  //  pr::CleanUp();
    return 0;
}
